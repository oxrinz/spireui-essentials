"use client"

import * as React from "react"
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from 'three'
import { useTexture } from "@react-three/drei";
import { GPUComputationRenderer } from "three/examples/jsm/Addons.js";
import { FluidSim } from "vexel-tools";

function Component(props: any) {
  const t = useThree();


  const logo: any = useTexture(props.src);
  const bubble = useTexture(props.particleSrc);

  const [logoWidth, logoHeight] = [Math.round(logo.source.data.width), Math.round(logo.source.data.height)]

  const [gpgpuWidth, gpgpuHeight] = [logoWidth * props.frequency, logoHeight * props.frequency]

  const pointsRef = useRef<any>()
  const imageRef = useRef<any>()

  const [prevMouse, setPrevMouse] = React.useState({ x: 0, y: 0 });
  const [mouseDelta, setMouseDelta] = React.useState({ x: 0, y: 0 });

  var settings = {
    simRes: 1 / 1000,
    curl: 0,
    dt: 0.016,
    iterations: 3,
    splatMultiplier: 100,

    densityDissipation: props.densityDissipation,
    velocityDissipation: props.velocityDissipation,
    pressureDissipation: props.pressureDissipation,
  }

  const fluidSim = React.useMemo(() => new FluidSim(t.gl.domElement.width, t.gl.domElement.height, t.gl, settings), [t, props])

  const compute: any = useMemo(() => {
    const compute = new GPUComputationRenderer(gpgpuWidth, gpgpuHeight, t.gl)

    const dtPosition: any = compute.createTexture()

    for (let i = 0; i < gpgpuWidth * gpgpuHeight; i++) {
      const i4 = i * 4
      dtPosition.image[i4 + 0] = Math.random()
      dtPosition.image[i4 + 1] = Math.random()
      dtPosition.image[i4 + 2] = Math.random()
      dtPosition.image[i4 + 3] = 1
    }

    const positionVariable = compute.addVariable(
      'texturePosition',
      simFrag,
      dtPosition
    )

    positionVariable.material.uniforms['time'] = { value: 0 }
    positionVariable.material.uniforms['logoTexture'] = { value: logo }
    positionVariable.material.uniforms['imageRes'] = { value: new THREE.Vector2(gpgpuWidth, gpgpuHeight) }
    positionVariable.material.uniforms['canvasRes'] = { value: new THREE.Vector2(0, 0) }
    positionVariable.material.uniforms['fluidTex'] = { value: null }
    positionVariable.material.uniforms['scale'] = { value: props.scale }

    positionVariable.wrapS = THREE.RepeatWrapping
    positionVariable.wrapT = THREE.RepeatWrapping

    compute.init()

    return compute
  }, [t, props])

  const particleUniforms = useMemo(
    () => ({
      time: { value: 0 },
      positionTexture: { value: null },
      logoTexture: { value: logo },
      imageRes: { value: new THREE.Vector2(logoWidth, logoHeight) },
      canvasRes: { value: new THREE.Vector4() },
      size: { value: props.particleSize },
      bubbleTex: { value: bubble },
      scale: { value: props.scale }
    }),
    []
  )

  const imageUniforms = useMemo(
    () => ({
      tex: { value: logo },
      imageRes: { value: new THREE.Vector2(logoWidth, logoHeight) },
      canvasRes: { value: new THREE.Vector4() },
      fluidTex: { value: null },
      bubbleTex: { value: bubble },
      scale: { value: props.scale }
    }),
    []
  )

  const particlesPosition = useMemo(() => {
    const array = new Float32Array(gpgpuWidth * gpgpuHeight * 3)
    const reference = new Float32Array(gpgpuWidth * gpgpuHeight * 2)

    for (let i = 0; i < gpgpuWidth * gpgpuHeight; i++) {
      let x = Math.random()
      let y = Math.random() * 10
      let z = Math.random()

      // --> uv
      let xx = (i % gpgpuWidth) / gpgpuWidth
      let yy = (i % gpgpuHeight) / gpgpuHeight

      array.set([x, y, z], i * 3)
      reference.set([xx, yy], i * 2)
    }

    return {
      positions: array,
      reference: reference
    }
  }, [])

  var splats: any = []

  useFrame((state) => {

    var pointer = new THREE.Vector2((state.pointer.x / props.scale + 1) / 2 * state.gl.domElement.width, (-state.pointer.y / props.scale + 1) / 2 * state.gl.domElement.height)

    setMouseDelta({ x: pointer.x - prevMouse.x, y: pointer.y - prevMouse.y });
    var mouseMultiplier = props.fluidForce;
    splats = [{ x: pointer.x, y: pointer.y, dx: mouseDelta.x / mouseMultiplier, dy: mouseDelta.y / mouseMultiplier, radius: 0.001 * props.mouseRadius }];

    state.gl.autoClear = false;

    fluidSim.compute(splats);
    state.gl.setRenderTarget(null);

    setPrevMouse(pointer);

    compute.compute()
    let elapseTime = state.clock.getElapsedTime()

    pointsRef.current.material.uniforms.positionTexture.value =
      compute.getCurrentRenderTarget(compute.variables[0]).texture

    compute.variables[0].material.uniforms.time.value = elapseTime

    pointsRef.current.material.uniforms.time.value = elapseTime

    var res = new THREE.Vector2(state.gl.domElement.width, state.gl.domElement.height);
    state.gl.getSize(res)

    pointsRef.current.material.uniforms.canvasRes.value = res
    pointsRef.current.material.uniforms.size.value = props.particleSize
    pointsRef.current.material.uniforms.scale.value = props.scale

    compute.variables[0].material.uniforms.canvasRes.value = res

    imageRef.current.material.uniforms.canvasRes.value = res
    imageRef.current.material.uniforms.scale.value = props.scale

    compute.variables[0].material.uniforms.scale.value = props.scale

    imageRef.current.material.uniforms.fluidTex.value = fluidSim.densityTexture
    compute.variables[0].material.uniforms.fluidTex.value = fluidSim.densityTexture
  })

  return (
    <group>
      <mesh
        ref={imageRef}>
        <planeGeometry args={[1, 1, 1, 1]} />
        <shaderMaterial
          fragmentShader={imageFrag}
          vertexShader={imageVert}
          uniforms={imageUniforms}
          transparent={true}
        />
      </mesh>
      <points
        ref={pointsRef}>
        <bufferGeometry>
          <bufferAttribute
            attach={'attributes-position'}
            itemSize={3}
            array={particlesPosition.positions}
            count={gpgpuWidth * gpgpuHeight}
          />
          <bufferAttribute
            attach={'attributes-reference'}
            itemSize={2}
            count={gpgpuWidth * gpgpuHeight * 2}
            array={particlesPosition.reference}
          />
        </bufferGeometry>
        <shaderMaterial
          vertexShader={pointVert}
          fragmentShader={pointFrag}
          uniforms={particleUniforms}
          blending={THREE.NormalBlending}
          transparent={true}
        />
      </points>
    </group>
  );

}

interface ComponentProps {
  src: string;
  particleSrc: string;
  frequency?: number;
  densityDissipation?: number;
  velocityDissipation?: number;
  pressureDissipation?: number;
  particleSize?: number;
  fluidForce?: number;
  mouseRadius?: number;
  scale?: number;

}

export default function ParticleImage({ src, particleSrc, frequency, densityDissipation, velocityDissipation, pressureDissipation, particleSize, fluidForce, mouseRadius, scale }: ComponentProps) {
  return (
    <Canvas className="h-screen">
      <Component src={src} particleSrc={particleSrc} frequency={frequency ?? 1} densityDissipation={densityDissipation ?? 0.99} velocityDissipation={velocityDissipation ?? 0.99} pressureDissipation={pressureDissipation ?? 0.99} particleSize={particleSize ?? 3} fluidForce={fluidForce ?? 50} mouseRadius={mouseRadius ?? 3} scale={scale ?? 0.8} />
    </Canvas>
  )
}

var pointVert
  = /* glsl */ `

precision highp float;

attribute vec2 reference;

varying vec2 imgUv;
varying vec2 vUv;
varying vec4 worldPos;

uniform sampler2D positionTexture;

uniform float size;


void main() {
  vUv = reference;
  imgUv = uv;
  vec3 pos = texture(positionTexture, reference).xyz;
  // pos = pos * position;

  vec4 mvPosition = vec4( pos , 1.);

  worldPos = mvPosition;

  gl_PointSize = size;

  gl_Position = mvPosition;
}
`;

var pointFrag
  = /* glsl */ `

precision highp float;

uniform sampler2D logoTexture;
uniform sampler2D fluidTex;
uniform sampler2D bubbleTex;

uniform vec2 imageRes;
uniform vec2 canvasRes;

uniform float scale;

varying vec2 vUv;
varying vec4 worldPos;

vec2 calcUv(vec2 uv)
{
  float canvasAspect = canvasRes.x / canvasRes.y;
  float imageAspect2 = imageRes.x / imageRes.y;

  vec2 scale;
  scale = vec2(imageAspect2 / canvasAspect, 1.0);

  if (canvasAspect > imageAspect2)
  {
    scale = vec2(1.0, (imageRes.y / imageRes.x) / (canvasRes.y / canvasRes.x));
  }

  float aspectMultiplier = (imageAspect2 / canvasAspect);

  if (aspectMultiplier < 1.0)
    {
      aspectMultiplier = (canvasAspect / imageAspect2);
    }

  return (uv - 0.5) / scale * aspectMultiplier + 0.5;

}

float minmax(float value)
{
  return max(min(value, 1.0), 0.0);
}

void main() {
    vec2 targetUv = (gl_FragCoord.xy / canvasRes - 0.5) / scale + 0.5;
    vec2 newUv = vUv;

    float spawnDistance =  minmax(distance(targetUv, newUv) * 10.0 - 0.0001);
    vec4 color = texture2D(logoTexture, calcUv(newUv)) * texture2D(bubbleTex, gl_PointCoord.xy);
    color = vec4(color.rgb, color.a * spawnDistance);


    gl_FragColor = color;
}
`;

var simFrag
  = /* glsl */ `

precision highp float;

// #region shit

//	--------------------------------------------------------------------
//	Optimized implementation of 3D/4D bitangent noise.
//	Based on stegu's simplex noise: https://github.com/stegu/webgl-noise.
//	Contact : atyuwen@gmail.com
//	Author : Yuwen Wu (https://atyuwen.github.io/)
//	License : Distributed under the MIT License.
//	--------------------------------------------------------------------

// Permuted congruential generator (only top 16 bits are well shuffled).
// References: 1. Mark Jarzynski and Marc Olano, "Hash Functions for GPU Rendering".
//             2. UnrealEngine/Random.ush. https://github.com/EpicGames/UnrealEngine
uvec2 _pcg3d16(uvec3 p)
{
	uvec3 v = p * 1664525u + 1013904223u;
	v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
	v.x += v.y*v.z; v.y += v.z*v.x;
	return v.xy;
}
uvec2 _pcg4d16(uvec4 p)
{
	uvec4 v = p * 1664525u + 1013904223u;
	v.x += v.y*v.w; v.y += v.z*v.x; v.z += v.x*v.y; v.w += v.y*v.z;
	v.x += v.y*v.w; v.y += v.z*v.x;
	return v.xy;
}

// Get random gradient from hash value.
vec3 _gradient3d(uint hash)
{
	vec3 g = vec3(uvec3(hash) & uvec3(0x80000, 0x40000, 0x20000));
	return g * (1.0 / vec3(0x40000, 0x20000, 0x10000)) - 1.0;
}
vec4 _gradient4d(uint hash)
{
	vec4 g = vec4(uvec4(hash) & uvec4(0x80000, 0x40000, 0x20000, 0x10000));
	return g * (1.0 / vec4(0x40000, 0x20000, 0x10000, 0x8000)) - 1.0;
}

// Optimized 3D Bitangent Noise. Approximately 113 instruction slots used.
// Assume p is in the range [-32768, 32767].
vec3 BitangentNoise3D(vec3 p)
{
	const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
	const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

	// First corner
	vec3 i = floor(p + dot(p, C.yyy));
	vec3 x0 = p - i + dot(i, C.xxx);

	// Other corners
	vec3 g = step(x0.yzx, x0.xyz);
	vec3 l = 1.0 - g;
	vec3 i1 = min(g.xyz, l.zxy);
	vec3 i2 = max(g.xyz, l.zxy);

	// x0 = x0 - 0.0 + 0.0 * C.xxx;
	// x1 = x0 - i1  + 1.0 * C.xxx;
	// x2 = x0 - i2  + 2.0 * C.xxx;
	// x3 = x0 - 1.0 + 3.0 * C.xxx;
	vec3 x1 = x0 - i1 + C.xxx;
	vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
	vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

	i = i + 32768.5;
	uvec2 hash0 = _pcg3d16(uvec3(i));
	uvec2 hash1 = _pcg3d16(uvec3(i + i1));
	uvec2 hash2 = _pcg3d16(uvec3(i + i2));
	uvec2 hash3 = _pcg3d16(uvec3(i + 1.0 ));

	vec3 p00 = _gradient3d(hash0.x); vec3 p01 = _gradient3d(hash0.y);
	vec3 p10 = _gradient3d(hash1.x); vec3 p11 = _gradient3d(hash1.y);
	vec3 p20 = _gradient3d(hash2.x); vec3 p21 = _gradient3d(hash2.y);
	vec3 p30 = _gradient3d(hash3.x); vec3 p31 = _gradient3d(hash3.y);

	// Calculate noise gradients.
	vec4 m = clamp(0.5 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0, 1.0);
	vec4 mt = m * m;
	vec4 m4 = mt * mt;

	mt = mt * m;
	vec4 pdotx = vec4(dot(p00, x0), dot(p10, x1), dot(p20, x2), dot(p30, x3));
	vec4 temp = mt * pdotx;
	vec3 gradient0 = -8.0 * (temp.x * x0 + temp.y * x1 + temp.z * x2 + temp.w * x3);
	gradient0 += m4.x * p00 + m4.y * p10 + m4.z * p20 + m4.w * p30;

	pdotx = vec4(dot(p01, x0), dot(p11, x1), dot(p21, x2), dot(p31, x3));
	temp = mt * pdotx;
	vec3 gradient1 = -8.0 * (temp.x * x0 + temp.y * x1 + temp.z * x2 + temp.w * x3);
	gradient1 += m4.x * p01 + m4.y * p11 + m4.z * p21 + m4.w * p31;

	// The cross products of two gradients is divergence free.
	return cross(gradient0, gradient1) * 3918.76;
}

// 4D Bitangent noise. Approximately 163 instruction slots used.
// Assume p is in the range [-32768, 32767].
vec3 BitangentNoise4D(vec4 p)
{
	const vec4 F4 = vec4( 0.309016994374947451 );
	const vec4  C = vec4( 0.138196601125011,  // (5 - sqrt(5))/20  G4
	                      0.276393202250021,  // 2 * G4
	                      0.414589803375032,  // 3 * G4
	                     -0.447213595499958 ); // -1 + 4 * G4

	// First corner
	vec4 i  = floor(p + dot(p, F4) );
	vec4 x0 = p -   i + dot(i, C.xxxx);

	// Other corners

	// Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
	vec4 i0;
	vec3 isX = step( x0.yzw, x0.xxx );
	vec3 isYZ = step( x0.zww, x0.yyz );
	// i0.x = dot( isX, vec3( 1.0 ) );
	i0.x = isX.x + isX.y + isX.z;
	i0.yzw = 1.0 - isX;
	// i0.y += dot( isYZ.xy, vec2( 1.0 ) );
	i0.y += isYZ.x + isYZ.y;
	i0.zw += 1.0 - isYZ.xy;
	i0.z += isYZ.z;
	i0.w += 1.0 - isYZ.z;

	// i0 now contains the unique values 0,1,2,3 in each channel
	vec4 i3 = clamp( i0, 0.0, 1.0 );
	vec4 i2 = clamp( i0 - 1.0, 0.0, 1.0 );
	vec4 i1 = clamp( i0 - 2.0, 0.0, 1.0 );

	// x0 = x0 - 0.0 + 0.0 * C.xxxx
	// x1 = x0 - i1  + 1.0 * C.xxxx
	// x2 = x0 - i2  + 2.0 * C.xxxx
	// x3 = x0 - i3  + 3.0 * C.xxxx
	// x4 = x0 - 1.0 + 4.0 * C.xxxx
	vec4 x1 = x0 - i1 + C.xxxx;
	vec4 x2 = x0 - i2 + C.yyyy;
	vec4 x3 = x0 - i3 + C.zzzz;
	vec4 x4 = x0 + C.wwww;

	i = i + 32768.5;
	uvec2 hash0 = _pcg4d16(uvec4(i));
	uvec2 hash1 = _pcg4d16(uvec4(i + i1));
	uvec2 hash2 = _pcg4d16(uvec4(i + i2));
	uvec2 hash3 = _pcg4d16(uvec4(i + i3));
	uvec2 hash4 = _pcg4d16(uvec4(i + 1.0 ));

	vec4 p00 = _gradient4d(hash0.x); vec4 p01 = _gradient4d(hash0.y);
	vec4 p10 = _gradient4d(hash1.x); vec4 p11 = _gradient4d(hash1.y);
	vec4 p20 = _gradient4d(hash2.x); vec4 p21 = _gradient4d(hash2.y);
	vec4 p30 = _gradient4d(hash3.x); vec4 p31 = _gradient4d(hash3.y);
	vec4 p40 = _gradient4d(hash4.x); vec4 p41 = _gradient4d(hash4.y);

	// Calculate noise gradients.
	vec3 m0 = clamp(0.6 - vec3(dot(x0, x0), dot(x1, x1), dot(x2, x2)), 0.0, 1.0);
	vec2 m1 = clamp(0.6 - vec2(dot(x3, x3), dot(x4, x4)             ), 0.0, 1.0);
	vec3 m02 = m0 * m0; vec3 m03 = m02 * m0;
	vec2 m12 = m1 * m1; vec2 m13 = m12 * m1;

	vec3 temp0 = m02 * vec3(dot(p00, x0), dot(p10, x1), dot(p20, x2));
	vec2 temp1 = m12 * vec2(dot(p30, x3), dot(p40, x4));
	vec4 grad0 = -6.0 * (temp0.x * x0 + temp0.y * x1 + temp0.z * x2 + temp1.x * x3 + temp1.y * x4);
	grad0 += m03.x * p00 + m03.y * p10 + m03.z * p20 + m13.x * p30 + m13.y * p40;

	temp0 = m02 * vec3(dot(p01, x0), dot(p11, x1), dot(p21, x2));
	temp1 = m12 * vec2(dot(p31, x3), dot(p41, x4));
	vec4 grad1 = -6.0 * (temp0.x * x0 + temp0.y * x1 + temp0.z * x2 + temp1.x * x3 + temp1.y * x4);
	grad1 += m03.x * p01 + m03.y * p11 + m03.z * p21 + m13.x * p31 + m13.y * p41;

	// The cross products of two gradients is divergence free.
	return cross(grad0.xyz, grad1.xyz) * 81.0;
}

// #endregion

uniform sampler2D texturePosition;
uniform sampler2D fluidTex;

uniform vec2 imageRes;
uniform vec2 canvasRes;

uniform float scale;

uniform float time;

void main() {
  vec2 uv = (gl_FragCoord.xy / imageRes - 0.5) * scale + 0.5;

  vec4 fluidNoise = vec4(BitangentNoise4D(vec4(uv * 100.0, 0.0, 0.0)), 0.0) / 10.0;
  vec4 fluid = texture2D(fluidTex, gl_FragCoord.xy / imageRes);
  vec2 fluidMultiplier = vec2(1.0) * (vec2(fluid.r - fluid.b, fluid.g - fluid.a) / 1000.0);

  float noiseDistance = distance((uv - 0.5) * 2.0 + fluidMultiplier, (uv - 0.5) * 2.0);

  gl_FragColor = vec4((uv - 0.5) * 2.0 + fluidMultiplier + fluidMultiplier * fluidNoise.xy, -1.0, 0.0);
}
`;

var imageFrag
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

uniform sampler2D tex;
uniform sampler2D fluidTex;

uniform vec2 imageRes;
uniform vec2 canvasRes;

uniform float scale;

varying vec2 vUv;



vec2 getScale()
{
  float canvasAspect = canvasRes.x / canvasRes.y;
  float imageAspect2 = imageRes.x / imageRes.y;

  vec2 scale;
  scale = vec2(imageAspect2 / canvasAspect, 1.0);

  if (canvasAspect < imageAspect2)
  {
    scale = vec2(1.0, (imageRes.y / imageRes.x) / (canvasRes.y / canvasRes.x));
  }


  return scale;

}

void main () {

    vec2 uv = (vUv - 0.5) / scale / getScale() + 0.5;

    vec2 newUv = (vUv - 0.5) / scale + 0.5;

    vec4 fluid = texture2D(fluidTex, newUv);
    vec2 densUv = newUv + vec2(1.0) * (-vec2(fluid.r - fluid.b, fluid.g - fluid.a) / 1000.0);
    vec2 fluidUv = (densUv - 0.5) / getScale() + 0.5;

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)
      {
        discard;
      }

    gl_FragColor = texture2D(tex, fluidUv) * vec4(1.0, 1.0, 1.0, 1.0 - distance(uv, fluidUv) * 10.0);
}
`;

var imageVert
  = /* glsl */ `

precision highp float;
varying vec2 vUv;

void main() {
    vUv = uv;
    gl_Position = vec4( position / vec3(0.5), 1.0 );
}
`;
