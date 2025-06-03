/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./public/*.{html,js}"],
  theme: {
    extend: {
      colors:{
        "color-primary":"001051e",
        "color-primary-light":"001051e",
        "color-primary-dark":"001051e",
        "color-secondary":"001051e",
        "color-gray":"#333",
        "color-white":"#ff",
        "color-blob":"001051e",
      }
    },
  },
  plugins: [],
}

