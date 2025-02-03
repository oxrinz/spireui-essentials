import * as React from "react"
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useMemo, useRef } from "react";
import * as THREE from 'three'
import { Stats, useTexture } from "@react-three/drei";
import { GPUComputationRenderer } from "three/examples/jsm/Addons.js";
import { Bloom, EffectComposer } from "@react-three/postprocessing";

function Component({ particleSrc, amount, gravity, brightness, spawnHeight, spawnWidth, lifeTime, noiseAmplitude, noiseFrequency, noiseEvolve, startColor, endColor, particleSize, mouseForce, mouseRadius, velocityFalloff }: any) {

  const t = useThree();

  const bubble = useTexture(particleSrc);

  const [gpgpuWidth, gpgpuHeight] = [amount, amount]

  const pointsRef = useRef<any>()
  const raycastRef = useRef<any>()

  const prevMousePos = useRef(new THREE.Vector2(0, 0))

  const compute: any = useMemo(() => {
    const compute = new GPUComputationRenderer(gpgpuWidth, gpgpuHeight, t.gl)

    const dtPosition: any = compute.createTexture()

    for (let i = 0; i < gpgpuWidth * gpgpuHeight; i++) {
      const i4 = i * 4
      dtPosition.image[i4 + 0] = Math.random()
      dtPosition.image[i4 + 1] = Math.random()
      dtPosition.image[i4 + 2] = Math.random()
      dtPosition.image[i4 + 3] = Math.random()
    }

    const velocityVariable = compute.addVariable(
      'textureVelocity',
      velSimFrag,
      dtPosition
    )

    const positionVariable = compute.addVariable(
      'texturePosition',
      simFrag,
      dtPosition
    )

    compute.setVariableDependencies(velocityVariable, [positionVariable, velocityVariable])
    compute.setVariableDependencies(positionVariable, [positionVariable, velocityVariable])

    positionVariable.material.uniforms['texturePosition'] = { value: null }
    positionVariable.material.uniforms['textureVelocity'] = { value: null }
    positionVariable.material.uniforms['time'] = { value: 0 }
    positionVariable.material.uniforms['canvasRes'] = { value: new THREE.Vector2(0, 0) }
    positionVariable.material.uniforms['textureRes'] = { value: new THREE.Vector2(gpgpuWidth, gpgpuHeight) }

    positionVariable.material.uniforms['brightness'] = { value: null }
    positionVariable.material.uniforms['spawnWidth'] = { value: 0 }
    positionVariable.material.uniforms['spawnHeight'] = { value: 0 }
    positionVariable.material.uniforms['lifeTime'] = { value: 0 }
    positionVariable.material.uniforms['gravity'] = { value: 0 }
    positionVariable.material.uniforms['noiseFrequency'] = { value: 0 }
    positionVariable.material.uniforms['noiseAmplitude'] = { value: 0 }
    positionVariable.material.uniforms['noiseEvolve'] = { value: 0 }
    positionVariable.material.uniforms['mousePos'] = { value: new THREE.Vector2(0, 0) }

    velocityVariable.material.uniforms['velocityFalloff'] = { value: null }
    velocityVariable.material.uniforms['texturePosition'] = { value: null }
    velocityVariable.material.uniforms['textureVelocity'] = { value: null }
    velocityVariable.material.uniforms['mouseForce'] = { value: 0 }
    velocityVariable.material.uniforms['mouseRadius'] = { value: 0 }
    velocityVariable.material.uniforms['mousePos'] = { value: new THREE.Vector2(0, 0) }
    velocityVariable.material.uniforms['mouseDelta'] = { value: new THREE.Vector2(0, 0) }
    velocityVariable.material.uniforms['textureRes'] = { value: new THREE.Vector2(gpgpuWidth, gpgpuHeight) }

    positionVariable.wrapS = THREE.RepeatWrapping
    positionVariable.wrapT = THREE.RepeatWrapping

    velocityVariable.wrapS = THREE.RepeatWrapping
    velocityVariable.wrapT = THREE.RepeatWrapping


    compute.init()

    return compute
  }, [])

  const particleUniforms = useMemo(
    () => ({
      time: { value: 0 },
      velocityTexture: { value: null },
      positionTexture: { value: null },
      canvasRes: { value: new THREE.Vector4() },
      size: { value: particleSize },
      brightness: { value: 0 },
      bubbleTex: { value: bubble },
      startColor: { value: null },
      endColor: { value: null }
    }),
    [particleSize]
  )

  const positions = new Float32Array(amount * amount * 3)
  const reference = new Float32Array(amount * amount * 2)

  for (let i = 0; i < amount * amount; i++) {

    let xx = (i % amount) / amount
    let yy = ~~(i / amount) / amount

    positions.set([0, 0, 0], i * 3)
    reference.set([xx, yy], i * 2)
  }


  useFrame((state) => {
    let elapseTime = state.clock.getElapsedTime()

    const raycaster = new THREE.Raycaster

    raycaster.setFromCamera(state.pointer, state.camera)

    const intersects = raycaster.intersectObject(raycastRef.current)

    var mousePos = new THREE.Vector2(0, 0)

    if (intersects.length > 0) {
      mousePos = new THREE.Vector2(intersects[0].point.x, intersects[0].point.y)
    }

    var res = new THREE.Vector2(state.gl.domElement.width, state.gl.domElement.height);
    state.gl.getSize(res)

    pointsRef.current.material.uniforms.canvasRes.value = res
    pointsRef.current.material.uniforms.velocityTexture.value = compute.getCurrentRenderTarget(compute.variables[0]).texture
    pointsRef.current.material.uniforms.positionTexture.value = compute.getCurrentRenderTarget(compute.variables[1]).texture
    pointsRef.current.material.uniforms.time.value = elapseTime
    pointsRef.current.material.uniforms.size.value = particleSize
    pointsRef.current.material.uniforms.brightness.value = brightness
    pointsRef.current.material.uniforms.startColor.value = new THREE.Vector3(startColor.rgb.r, startColor.rgb.g, startColor.rgb.b)
    pointsRef.current.material.uniforms.endColor.value = new THREE.Vector3(endColor.rgb.r, endColor.rgb.g, endColor.rgb.b)

    compute.variables[1].material.uniforms.time.value = elapseTime
    compute.variables[1].material.uniforms.canvasRes.value = res
    compute.variables[1].material.uniforms.brightness.value = brightness
    compute.variables[1].material.uniforms.spawnWidth.value = spawnWidth
    compute.variables[1].material.uniforms.spawnHeight.value = spawnHeight
    compute.variables[1].material.uniforms.lifeTime.value = lifeTime
    compute.variables[1].material.uniforms.gravity.value = gravity
    compute.variables[1].material.uniforms.noiseAmplitude.value = noiseAmplitude
    compute.variables[1].material.uniforms.noiseFrequency.value = noiseFrequency
    compute.variables[1].material.uniforms.noiseEvolve.value = noiseEvolve

    compute.variables[0].material.uniforms.mouseForce.value = mouseForce
    compute.variables[0].material.uniforms.mouseRadius.value = mouseRadius
    compute.variables[0].material.uniforms.mousePos.value = mousePos
    compute.variables[0].material.uniforms.velocityFalloff.value = velocityFalloff
    const mouseDelta = mousePos.clone()
    mouseDelta.sub(prevMousePos.current)
    console.log(mouseDelta)
    compute.variables[0].material.uniforms.mouseDelta.value = mouseDelta

    state.gl.autoClear = false;

    state.gl.setRenderTarget(null);
    compute.compute()

    prevMousePos.current = mousePos
  })

  return (
    <group>
      <mesh rotation={[0, 1.570796325 * 2, 0]} ref={raycastRef}>
        <planeGeometry args={[10, 10]} />
        <meshBasicMaterial visible={false} />
      </mesh>
      <points
        frustumCulled={false}
        ref={pointsRef}>
        <bufferGeometry>
          <bufferAttribute
            attach={'attributes-position'}
            itemSize={3}
            array={positions}
            count={gpgpuWidth * gpgpuHeight}
          />
          <bufferAttribute
            attach={'attributes-reference'}
            itemSize={2}
            count={gpgpuWidth * gpgpuHeight * 2}
            array={reference}
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

interface ParticleSeaProps {
  particleSrc: string
  amount?: number
  particleSize?: number
  brightness?: number
  spawnWidth?: number
  spawnHeight?: number
  lifeTime?: number
  gravity?: number
  noiseFrequency?: number
  noiseAmplitude?: number
  noiseEvolve?: number
  startColor?: any
  endColor?: any
  mouseRadius?: number
  mouseForce?: number
  velocityFalloff?: number
}

export default function ParticleSea(props: ParticleSeaProps) {
  return (
    <Canvas camera={{ position: [0, 0, -1] }} className="h-screen">
      <EffectComposer>
        <Bloom mipmapBlur luminanceThreshold={1} intensity={1} />
      </EffectComposer>
      <Component particleSrc={props.particleSrc} amount={props.amount ?? 50} particleSize={props.particleSize ?? 1} brightness={props.brightness ?? 0.5} spawnWidth={props.spawnWidth ?? 5} spawnHeight={props.spawnHeight ?? 5} lifeTime={props.lifeTime ?? 0.999} gravity={props.gravity ?? 0.4} noiseFrequency={props.noiseFrequency ?? 3} noiseAmplitude={props.noiseAmplitude ?? 1} noiseEvolve={props.noiseEvolve ?? 10} startColor={props.startColor ?? { rgb: { r: 255, g: 255, b: 255 } }} endColor={props.endColor ?? { rgb: { r: 255, g: 255, b: 255 } }} mouseForce={props.mouseForce ?? 0.05} mouseRadius={props.mouseRadius ?? 0.1} velocityFalloff={props.velocityFalloff ?? 0.01} />
      <Stats />
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

uniform vec2 canvasRes;

uniform sampler2D positionTexture;

uniform float size;

float rand(vec2 co){
  return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {

  vUv = reference;
  imgUv = uv;
  vec3 pos = texture(positionTexture, reference).xyz;

  vec4 mvPosition = projectionMatrix * modelViewMatrix * (vec4(pos, 1.0));

  worldPos = mvPosition;

  gl_PointSize = size;

  gl_Position = mvPosition;
}
`;

var pointFrag
  = /* glsl */ `

precision highp float;

uniform sampler2D bubbleTex;
uniform sampler2D positionTexture;
uniform sampler2D velocityTexture;

uniform vec2 canvasRes;

uniform float size;
uniform float brightness;
uniform vec3 startColor;
uniform vec3 endColor;

uniform bool contain;

varying vec2 vUv;
varying vec4 worldPos;

void main() {
  float progress = texture2D(positionTexture, vUv).a;
  float progressAlpha = 1.0 - abs(progress * 2.0 - 1.0);
  vec4 colorMult = vec4(mix(startColor, endColor, progress), 1.0);
  vec4 color = texture2D(bubbleTex, gl_PointCoord.xy) * colorMult;

  gl_FragColor = color * vec4(brightness) * vec4(progressAlpha);
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

uniform vec2 textureRes;
uniform vec2 canvasRes;

uniform float spawnWidth;
uniform float spawnHeight;
uniform float lifeTime;
uniform float gravity;
uniform float noiseFrequency;
uniform float noiseAmplitude;
uniform float noiseEvolve;
uniform float mouseForce;
uniform float mouseRadius;

uniform float time;

float rand(vec2 co){
  return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
  vec4 prevPosition = texture2D(texturePosition, (gl_FragCoord.xy) / textureRes);
  vec4 prevVelocity = texture2D(textureVelocity, (gl_FragCoord.xy) / textureRes);

  vec3 noise = BitangentNoise3D(vec3(prevPosition.xyz + time * (noiseEvolve / 100.0)) * noiseFrequency) * (noiseAmplitude / 10000.0);
  vec3 grav = vec3(1.0, 1.0 + gravity / 100.0, 1.0);
  vec3 position = (prevPosition.xyz + noise) * grav;

  if (prevPosition.a <= 0.0)
  {
    prevPosition.a = 1.0;

    vec3 spawnPoint = vec3((rand((gl_FragCoord.xy) + time) - 0.5) * spawnWidth, (rand(gl_FragCoord.yx * 10.0 + time) - 0.5) * spawnHeight, 0.0);
    vec3 randomSpawnOffset = (vec3(rand(gl_FragCoord.xy * 10.0), rand(gl_FragCoord.xy * 100.0), rand(gl_FragCoord.xy)) - 0.5) / 10.0;
    position = randomSpawnOffset + spawnPoint;
  }


  gl_FragColor = vec4(position + prevVelocity.xyz, prevPosition.a - clamp(1.0 - lifeTime, 0.0, 1.0) * (rand((gl_FragCoord.xy)) * 3.0));
}
`;

var velSimFrag
  = /* glsl */ `

precision highp float;

uniform vec2 textureRes;
uniform vec2 canvasRes;

uniform float noiseEvolve;
uniform float mouseForce;
uniform float mouseRadius;
uniform float velocityFalloff;

uniform vec2 mousePos;
uniform vec2 mouseDelta;
uniform float time;


void main() {
  vec4 prevPosition = texture2D(texturePosition, (gl_FragCoord.xy) / textureRes);
  vec4 prevVelocity = texture2D(textureVelocity, (gl_FragCoord.xy) / textureRes);

  vec2 newVelocity = vec2(0.0);

  if (distance(mousePos, prevPosition.xy) < mouseRadius)
    {
        newVelocity = mouseDelta * mouseForce;
    }

  prevVelocity = vec4(mix(vec4(0.0), prevVelocity, step(0.0, prevPosition.a)));
  newVelocity = vec2(mix(vec2(0.0), newVelocity, step(0.0, prevPosition.a)));

  gl_FragColor = vec4(newVelocity + prevVelocity.xy * (1.0 - velocityFalloff), 0.0, 1.0);
}
`;
