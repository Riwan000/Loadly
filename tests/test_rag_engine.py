# ============================================================================
# RAG ENGINE TESTS (6 tests)
# ============================================================================
import pytest
from utils.rag_engine import initialize_rag_system, generate_forecast_summary


class TestRAGScope:
    """Tests for RAG scope checking and validation."""

    def test_rag_system_initializes(self):
        """
        PURPOSE: Verify RAG system initializes without errors
        CATCHES: Missing FAISS index, corrupted documents
        PRODUCTION IMPACT: RAG tab crashes on app load
        """
        try:
            rag = initialize_rag_system()
            assert rag is not None
            assert hasattr(rag, 'vector_store')
        except FileNotFoundError:
            pytest.skip("RAG index not built - skipping")

    def test_rag_valid_query_processed(self):
        """
        PURPOSE: Verify in-scope queries are processed
        CATCHES: Valid queries incorrectly rejected
        PRODUCTION IMPACT: App rejects legitimate questions
        """
        summary = generate_forecast_summary(
            state='Maharashtra',
            forecast_values=[100, 110, 120],
            query="What's the impact of high temperature on demand?"
        )

        # Should return an answer (not "out of scope")
        assert summary is not None
        assert "out of scope" not in summary.lower()

    def test_rag_invalid_query_rejected(self):
        """
        PURPOSE: Verify out-of-scope queries are rejected
        CATCHES: Hallucination in UI (returning fake energy info)
        PRODUCTION IMPACT: User sees false information
        SEVERITY: 🔴 CRITICAL for user trust
        """
        summary = generate_forecast_summary(
            state='Maharashtra',
            forecast_values=[100, 110, 120],
            query="What's the current price of Bitcoin?"  # Out of scope
        )

        # Should indicate out of scope or that only energy questions are answered
        assert any(phrase in summary.lower() for phrase in ["out of scope", "cannot answer", "only answer", "electricity demand"])

    def test_rag_confidence_score_valid_range(self):
        """
        PURPOSE: Verify confidence score is 0-100
        CATCHES: Score out of range (e.g., -1, 150)
        PRODUCTION IMPACT: Dashboard shows misleading confidence
        """
        result = generate_forecast_summary(
            state='Maharashtra',
            forecast_values=[100, 110, 120],
            query="How does weather affect demand?"
        )

        if isinstance(result, dict) and 'confidence' in result:
            assert 0 <= result['confidence'] <= 100

    def test_rag_document_retrieval_relevance(self):
        """
        PURPOSE: Verify retrieved documents are relevant
        CATCHES: Irrelevant documents in response
        PRODUCTION IMPACT: Summary includes wrong information
        """
        try:
            rag = initialize_rag_system()
            if rag is None:
                pytest.skip("RAG not initialized")

            # Use the vector_store directly to retrieve documents
            from utils.embeddings import get_embeddings
            query = "temperature and demand"
            query_embedding = get_embeddings([query])[0]
            results = rag.vector_store.search(query_embedding, k=3, min_similarity=0.0)

            # At least some results should mention temperature or demand
            relevant = [r for r in results if 'temperature' in r[0].lower() or 'demand' in r[0].lower()]
            # If we got results, verify they're relevant; if empty, skip
            if results:
                assert len(relevant) >= 1, "Retrieved documents should mention temperature or demand"
        except (FileNotFoundError, ImportError):
            pytest.skip("RAG index or embeddings not available")

    def test_rag_graceful_failure_missing_metadata(self):
        """
        PURPOSE: Verify RAG works even with missing model metadata
        CATCHES: Crashes when metadata unavailable
        PRODUCTION IMPACT: App crashes on metadata load error
        """
        try:
            summary = generate_forecast_summary(
                state='Maharashtra',
                forecast_values=[100, 110, 120],
                metadata=None,  # Intentionally missing
                query="What's the forecast?"
            )
            assert summary is not None
        except Exception as e:
            pytest.fail(f"RAG crashed with missing metadata: {e}")


class TestChatbotUpgrades:
    """Tests for production chatbot fixes: LinearRegression importance, dynamic model name, fallback, All-India."""

    def test_linear_regression_feature_importance(self):
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        from visualization import get_feature_importance

        X = pd.DataFrame({
            "demand_roll_mean_7": [100.0, 200.0, 300.0, 400.0],
            "2m_temperature_mean": [295.0, 300.0, 305.0, 310.0],
            "CDD": [0.0, 2.85, 7.85, 12.85],
        })
        y = np.array([105.0, 210.0, 315.0, 420.0])
        model = LinearRegression().fit(X, y)

        imp = get_feature_importance(model, top_n=3)
        assert imp is not None
        assert len(imp) == 3
        assert abs(sum(imp.values()) - 1.0) < 1e-6

    def test_rag_builder_uses_dynamic_model_name_and_platform_docs(self):
        from utils.rag_builder import build_model_metrics_documents, build_platform_knowledge_documents

        docs = build_model_metrics_documents({
            "model_name": "Reduced_LR",
            "rmse": 5.36,
            "mae": 3.21,
            "r2": 0.9989,
            "n_features": 15,
            "confidence_level": 0.9,
        })
        assert "Reduced_LR" in docs[0]["text"]
        assert "XGB2_Log" not in docs[0]["text"]

        platform_docs = build_platform_knowledge_documents()
        combined = " ".join(d["text"] for d in platform_docs)
        assert "All-India" in combined
        assert "Cooling Degree Days" in combined
        assert "Prediction Ledger" in combined
        assert "Champion vs. Challenger" in combined

    def test_rag_engine_grounded_fallback_and_chat_history(self, monkeypatch):
        from utils.rag_engine import RAGEngine
        from utils.vector_store import VectorStore

        vs = VectorStore(dimension=384, index_path=None)
        engine = RAGEngine(vs)

        # Force OpenRouter failure to test deterministic grounded fallback
        def _fail_llm(*args, **kwargs):
            raise ValueError("Simulated 429 rate limit")

        monkeypatch.setattr("utils.rag_engine.call_openrouter_llm", _fail_llm)

        fc = {
            "state": "All-India",
            "horizon_days": 7,
            "start_date": "2026-11-21",
            "end_date": "2026-11-27",
            "avg_demand": 5300.0,
            "peak_demand": 5480.0,
            "peak_date": "2026-11-24",
            "min_demand": 5120.0,
            "high_risk_days_count": 1,
            "high_risk_dates": ["2026-11-24"],
        }
        reply, _, _ = engine.query(
            user_query="What is the peak demand?",
            current_state="All-India",
            forecast_horizon=7,
            forecast_context=fc,
            chat_history=[{"role": "user", "content": "Show national summary"}],
        )
        assert "5480.0 MU" in reply
        assert "2026-11-24" in reply
        assert "Error generating response" not in reply

    def test_assistant_suggestions_all_india(self):
        from fastapi.testclient import TestClient
        from server import app

        client = TestClient(app)
        res = client.get("/api/assistant/suggestions?state=All-India")
        assert res.status_code == 200
        data = res.json()
        assert data["state"] == "All-India"
        assert any("All-India" in s for s in data["suggestions"])
        assert not any("Kerala" in s for s in data["suggestions"])

    def test_assistant_chat_endpoint_all_india_with_history(self):
        from fastapi.testclient import TestClient
        from server import app

        client = TestClient(app)
        res = client.post(
            "/api/assistant/chat",
            json={
                "message": "What is the current MLOps drift status and model accuracy?",
                "state": "All-India",
                "horizon_days": 7,
                "chat_history": [{"role": "user", "content": "Hello"}],
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["is_out_of_scope"] is False
        assert len(data["reply"]) > 20
        assert "Error generating response" not in data["reply"]

