import React, { useState } from 'react';
import { Key } from 'lucide-react';

/**
 * Inline operator key prompt shown inside a modal when an admin action needs a key.
 *
 * @param {{ message: string, onSubmit: (key: string) => void, onCancel: () => void }} props
 */
export default function AdminKeyPrompt({ message, onSubmit, onCancel }) {
  const [value, setValue] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!value.trim()) return;
    onSubmit(value);
    setValue('');
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="p-3 text-xs bg-amber-50 dark:bg-amber-950/30 border border-amber-300 dark:border-amber-800 text-amber-800 dark:text-amber-300 animate-fadeIn"
    >
      <div className="font-bold flex items-center gap-1.5">
        <Key className="w-4 h-4" />
        Operator authorization required
      </div>
      <div className="mt-1 opacity-90">{message}</div>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <input
          type="password"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          autoComplete="off"
          autoFocus
          spellCheck={false}
          aria-label="Operator admin key"
          placeholder="Admin key"
          className="flex-1 min-w-[180px] px-2 py-1.5 text-xs bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 text-slate-900 dark:text-slate-100 rounded-none focus:outline-none focus:ring-1 focus:ring-amber-500"
        />
        <button
          type="submit"
          disabled={!value.trim()}
          className="px-3 py-1.5 text-xs font-semibold bg-emerald-600 hover:bg-emerald-700 text-white rounded-none transition-colors disabled:opacity-50"
        >
          Unlock &amp; Continue
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="px-3 py-1.5 text-xs font-semibold bg-slate-200 hover:bg-slate-300 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 rounded-none transition-colors"
        >
          Cancel
        </button>
      </div>
      <div className="mt-1.5 text-[10px] opacity-75">
        Kept for this browser tab only (session storage) and sent as the X-Admin-Key header.
      </div>
    </form>
  );
}
