/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        parchment: {
          50: '#FFFBF5',
          100: '#FFF8EE',
          200: '#FAF0E6',
          300: '#F5E6D3',
          400: '#EDD5B3',
          500: '#E6C79C',
        },
        saffron: {
          50: '#FFF9E6',
          100: '#FFF0CC',
          200: '#FFE699',
          300: '#FFD966',
          400: '#F4C430',
          500: '#D2691E',
          600: '#A0522D',
        },
        heritage: {
          50: '#F5F1ED',
          100: '#E8DFD6',
          200: '#D4C4B0',
          300: '#B39A7E',
          400: '#8B7355',
          500: '#5A3E1B',
          600: '#3E2A14',
          700: '#2C1810',
        },
      },
      fontFamily: {
        serif: ['Crimson Text', 'Georgia', 'serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
        devanagari: ['Noto Serif Devanagari', 'serif'],
        kannada: ['Noto Sans Kannada', 'sans-serif'],
      },
      boxShadow: {
        'heritage': '0 4px 20px rgba(90, 62, 27, 0.1)',
        'heritage-lg': '0 8px 30px rgba(90, 62, 27, 0.15)',
      },
    },
  },
  plugins: [],
}

