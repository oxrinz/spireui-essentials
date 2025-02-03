import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import glsl from 'rollup-plugin-glsl'
import { resolve } from 'path'

export default defineConfig({
  plugins: [
    react(),
    glsl({
      include: '**/*.{glsl,vert,frag}',
      exclude: 'node_modules/**',
      sourceMap: true
    })
  ],
  root: './dev',
  resolve: {
    alias: {
      'your-lib': resolve(__dirname, './dist')
    }
  }
})