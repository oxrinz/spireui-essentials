import { defineConfig } from 'tsup'
import { glslifyPlugin } from './scripts/glsl-plugin'

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['cjs', 'esm'],
  dts: true,
  splitting: false,
  sourcemap: true,
  clean: true,
  treeshake: true,
  esbuildPlugins: [glslifyPlugin]
})