// scripts/glslify-plugin.ts
import { Plugin } from 'esbuild'
import glslify from 'glslify'
import fs from 'fs/promises'

export const glslifyPlugin: Plugin = {
  name: 'glslify',
  setup(build) {
    build.onLoad({ filter: /\.(glsl|vert|frag)$/ }, async (args) => {
      const source = await fs.readFile(args.path, 'utf8')
      const processed = glslify.compile(source, {
        basedir: process.cwd()
      })
      
      return {
        contents: `export default ${JSON.stringify(processed)}`,
        loader: 'js'
      }
    })
  }
}