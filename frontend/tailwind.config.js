/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        executive: {
          // Daylight / Common
          navy: '#1E3A8A',
          orange: '#EA580C',
          crimson: '#E11D48',
          slate: '#64748B',
          bgLight: '#F8FAFC',
          cardLight: '#FFFFFF',
          borderLight: '#E2E8F0',
          textLight: '#0F172A',
          mutedLight: '#475569',

          // Midnight (Dark)
          cyan: '#38BDF8',
          gold: '#FFB703',
          coral: '#FF4D6D',
          bgDark: '#0A0F1D',
          cardDark: '#131C31',
          borderDark: '#233050',
          textDark: '#F8FAFC',
          mutedDark: '#94A3B8',
        }
      }
    },
  },
  plugins: [],
}
