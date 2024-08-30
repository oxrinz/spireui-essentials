import * as React from "react"

import { Canvas, useFrame } from '@react-three/fiber'
import {  useTexture } from '@react-three/drei' 
import * as THREE from 'three'

function lerp(a: number, b: number, alpha: number) {
  return a + alpha * (b - a)
}

function Component({ imageData, index, frequency, density, speed }: any) {
  const ref = React.useRef<any>()


  var textures: any = [];
  var resolutions: any = [];

  for (let i = 0; i < imageData.length; i++) {
    textures.push(useTexture(imageData[i]))
    textures[i].wrapS = THREE.ClampToEdgeWrapping
    textures[i].wrapT = THREE.ClampToEdgeWrapping
    resolutions.push(new THREE.Vector2(textures[i].source.data.width, textures[i].source.data.height))
  }

  var uniforms = React.useMemo(() => ({
    canvasRes: { value: new THREE.Vector2(0, 0) },
    imageRes1: { value: new THREE.Vector2(0, 0) },
    imageRes2: { value: new THREE.Vector2(0, 0) },
    frequency: { value: frequency },
    density: { value: density },
    target: { value: 0 },
    tex1: { value: null },
    tex2: { value: null },
  }), [],)

  useFrame((state) => {
    var setIndex = index;
    setIndex = Math.min(setIndex, index);
    if (index > textures.length - 1) {
      setIndex = textures.length - 1;
    }
     else if (index < 0.0) {
      setIndex = 0
    }

    setIndex -= 0.0001

    // Settings
    ref.current.material.uniforms.frequency.value = new THREE.Vector2(frequency[0], frequency[1]);
    ref.current.material.uniforms.density.value = density;


    var targetValue = ref.current.material.uniforms['target'].value
    var value = lerp(targetValue, setIndex, 0.03 * speed);

    ref.current.material.uniforms.target.value = value;
    ref.current.material.needsupdate = true;

    var floor = Math.floor(value);
    var ceil = Math.ceil(value);

    ref.current.material.uniforms.tex1.value = textures[floor]
    ref.current.material.uniforms.tex2.value = textures[ceil]

    var res = new THREE.Vector2(0.0, 0.0);
    state.gl.getSize(res)


    if (resolutions[floor]) {
      ref.current.material.uniforms.imageRes1.value = resolutions[floor]
    }
    ref.current.material.uniforms.imageRes2.value = resolutions[ceil]
    ref.current.material.uniforms.canvasRes.value = res
  })


  return (
    <mesh
      ref={ref}>
      <planeGeometry args={[1, 1, 1, 1]} />
      <shaderMaterial
        fragmentShader={fragmentShader}
        vertexShader={vertexShader}
        uniforms={uniforms}
      />
    </mesh>
  )
}

interface ComponentTypes {
  imageData: string[],
  speed?: number,
  density?: number,
  frequency?: number[],
  index?: number

}

export default function ImageCarousel({ imageData, index, frequency, density, speed }: ComponentTypes) {
  return (
    <Canvas style={{ width: "100%", height: "100%" }} >
      <Component imageData={imageData} speed={speed ?? 1} density={density ?? 0.5} frequency={frequency ?? [0.1, 25]} index={index ?? 0} />
    </Canvas>
  )
}

var fragmentShader

  = /* glsl */ `

precision highp float;
precision highp sampler2D;

//#region noise shit
vec3 mod289(vec3 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 mod289(vec4 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x) {
     return mod289(((x*34.0)+10.0)*x);
}

vec4 taylorInvSqrt(vec4 r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

float snoise(vec3 v)
  {
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 =   v - i + dot(i, C.xxx) ;

// Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  //   x0 = x0 - 0.0 + 0.0 * C.xxx;
  //   x1 = x0 - i1  + 1.0 * C.xxx;
  //   x2 = x0 - i2  + 2.0 * C.xxx;
  //   x3 = x0 - 1.0 + 3.0 * C.xxx;
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
  vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

// Permutations
  i = mod289(i);
  vec4 p = permute( permute( permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

// Gradients: 7x7 points over a square, mapped onto an octahedron.
// The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
  float n_ = 0.142857142857; // 1.0/7.0
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
  //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

// Mix final noise value
  vec4 m = max(0.5 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 105.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
                                dot(p2,x2), dot(p3,x3) ) );
  }
//#endregion

uniform sampler2D tex1;
uniform sampler2D tex2;

uniform vec2 imageRes1;
uniform vec2 imageRes2;

uniform vec2 canvasRes;

uniform vec2 frequency;
uniform float density;

uniform float target;

varying vec2 vUv;

float value = 1.0;
float rgbSeperate = 0.05;

float minmax(float noiseInput)
{
  return max(min(noiseInput, 1.0), 0.0);
}

float calcNoise(float flooredTarget)
{
  return snoise(vec3(vUv * frequency, flooredTarget));
}

float gradient(float corTarget)
{
  return (vUv.x + (corTarget) - 1.0) * density + 0.5;
}

float map(float value, float min1, float max1, float min2, float max2) {
  return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}

vec2 calcUv(vec2 uv)
{
  float canvasAspect = canvasRes.x / canvasRes.y;
  float imageAspect2 = uv.x / uv.y;

  vec2 scale;
  scale = vec2(imageAspect2 / canvasAspect, 1.0);

  if (canvasAspect > imageAspect2)
  {
    scale = vec2(1.0, (uv.y / uv.x) / (canvasRes.y / canvasRes.x));
  }

  return (vUv - 0.5) / scale + 0.5;

}

void main () {
    float corTarget = sin(target-floor(target)) / 0.841470984268;
    float flooredTarget = floor(target);

    vec2 uv1 = calcUv(imageRes1);
    vec2 uv2 = calcUv(imageRes2);

    vec4 color1 = texture2D(tex1, uv1);
    vec4 color2 = texture2D(tex2, uv2);

    float mixValue = round(minmax((gradient(corTarget) + calcNoise(flooredTarget)) + mix(-1.0, 1.0, corTarget)));

    gl_FragColor =  mix(color1, color2, mixValue);
}
`;

var vertexShader
  = /* glsl */ `

precision highp float;
varying vec2 vUv;

void main() {
    vUv = uv;
    gl_Position = vec4( position / vec3(0.5), 1.0 );
}
`;
