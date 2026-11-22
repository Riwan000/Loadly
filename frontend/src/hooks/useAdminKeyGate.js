import { useCallback, useState } from 'react';
import { adminKeyStore, AdminAuthError, ADMIN_ERROR } from '../services/api';

const DEFAULT_PROMPT_MESSAGE = 'Enter the operator admin key to run this action.';

/**
 * Gate for operator-only actions. Opens an inline key prompt when no key is stored
 * or the server rejects the stored key, then re-runs the action once a key is entered.
 *
 * @returns {{
 *   prompt: { message: string, retry: (() => void) | null } | null,
 *   hasKey: boolean,
 *   ensureKey: (retry: () => void) => boolean,
 *   handleAuthError: (err: unknown, retry: () => void) => boolean,
 *   submitKey: (key: string) => void,
 *   cancelPrompt: () => void,
 *   forgetKey: () => void
 * }}
 */
export function useAdminKeyGate() {
  const [prompt, setPrompt] = useState(null);
  const [hasKey, setHasKey] = useState(() => Boolean(adminKeyStore.get()));

  /** Returns true when a key is available; otherwise opens the prompt and returns false. */
  const ensureKey = useCallback((retry) => {
    if (adminKeyStore.get()) return true;
    setHasKey(false);
    setPrompt({ message: DEFAULT_PROMPT_MESSAGE, retry });
    return false;
  }, []);

  /** Returns true when the error was an auth problem the prompt now handles. */
  const handleAuthError = useCallback((err, retry) => {
    if (!(err instanceof AdminAuthError) || err.code === ADMIN_ERROR.DISABLED) return false;
    setHasKey(false);
    setPrompt({ message: err.message, retry });
    return true;
  }, []);

  const submitKey = useCallback((key) => {
    const trimmed = (key || '').trim();
    if (!trimmed) return;
    adminKeyStore.set(trimmed);
    setHasKey(true);
    const retry = prompt?.retry;
    setPrompt(null);
    if (retry) retry();
  }, [prompt]);

  const cancelPrompt = useCallback(() => setPrompt(null), []);

  const forgetKey = useCallback(() => {
    adminKeyStore.clear();
    setHasKey(false);
  }, []);

  return { prompt, hasKey, ensureKey, handleAuthError, submitKey, cancelPrompt, forgetKey };
}
