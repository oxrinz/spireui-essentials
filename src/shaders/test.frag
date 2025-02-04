uniform float time;
varying vec2 vUv;

#pragma glslify: snoise3 = require(glsl-noise/simplex/3d)

void main() {
  vec2 uv = vUv;
  
  float noise = snoise3(vec3(uv * 3.0, time * 0.2));
  
  vec3 color = 0.5 + 0.5 * cos(time + uv.xyx + vec3(0,2,4));
  color *= noise * 0.5 + 0.5;
  
  gl_FragColor = vec4(color, 1.0);
}