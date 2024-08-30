"use client"

import { Canvas, useFrame } from "@react-three/fiber";
import * as React from "react"
import { Vector2, Vector3 } from "three";

function Component(props: any) {
	const ref = React.useRef<any>()


	const uniforms = React.useMemo(
		() => ({
			color1: { value: null },
			color2: { value: null },
			color3: { value: null },
			color4: { value: null },
			time: { value: 0 },
			canvasRes: { value: new Vector2(0, 0) },
			frequency: { value: new Vector2(0, 0) },
		}),
		[]
	)

	useFrame((state) => {
		ref.current.material.uniforms.color1.value = new Vector3(props.color1.rgb.r, props.color1.rgb.g, props.color1.rgb.b)
		ref.current.material.uniforms.color2.value = new Vector3(props.color2.rgb.r, props.color2.rgb.g, props.color2.rgb.b)
		ref.current.material.uniforms.color3.value = new Vector3(props.color3.rgb.r, props.color3.rgb.g, props.color3.rgb.b)
		ref.current.material.uniforms.color4.value = new Vector3(props.color4.rgb.r, props.color4.rgb.g, props.color4.rgb.b)

		ref.current.material.uniforms.time.value = state.clock.getElapsedTime()
		ref.current.material.uniforms.canvasRes.value = new Vector2(state.gl.domElement.width, state.gl.domElement.height)
	})

	return (
		<mesh
			ref={ref}>
			<planeGeometry args={[1, 1, 1, 1]} />
			<shaderMaterial
				fragmentShader={frag}
				vertexShader={vert}
				uniforms={uniforms}
				transparent={true}
			/>
		</mesh>
	)
}

interface ComponentTypes {
	color1: { rgb: { r: number, g: number, b: number } },
	color2: { rgb: { r: number, g: number, b: number } },
	color3: { rgb: { r: number, g: number, b: number } },
	color4: { rgb: { r: number, g: number, b: number } },

}

export default function GradientBackground(props: ComponentTypes) {
	return (
		<Canvas className="h-screen">
			<Component {...props} />
		</Canvas>
	)
}


var frag
	= /* glsl */ `

precision highp float;
precision highp sampler2D;

// #region shit

float rand(vec2 co){
  return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

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

varying vec2 vUv;

uniform vec3 color1;
uniform vec3 color2;
uniform vec3 color3;
uniform vec3 color4;

uniform float time;
uniform vec2 canvasRes;

struct Light {
  vec2 position;
  vec3 color;
  float radius;
};

float testSeed = 12.0;

void main () {
  vec2 uv = vUv;

  float canvasAspect = canvasRes.x / canvasRes.y;

  // light gradient stuff
  float radiusMultiplier = BitangentNoise3D(vec3(uv * 10.0, 0.0) ).x * 12.0;

  Light lights[30];

  vec3 color = vec3(0.0, 0.0, 0.0);

  for(int i = 0; i < lights.length(); i++) {
    // #region light gen
    int lightColorValue = int(floor(rand(vec2(0.0, float(i))) * 4.0));
    vec3 lightColor;

    if (lightColorValue == 0) {
      lightColor = color1;
    }
    if (lightColorValue == 1) {
      lightColor = color2;
    }
    if (lightColorValue == 2) {
      lightColor = color3;
    }
    if (lightColorValue == 3) {
      lightColor = color4;
    }
    // #endregion

    float xOffsetMultiplier = (0.5 - rand(vec2(float(i), 1.0 + testSeed))) * 4.5;
    float yOffsetMultiplier = (0.5 - rand(vec2(float(i), 0.0 + testSeed))) * 4.5;

    float xTimeMultiplier = (0.5 - rand(vec2(float(i), 1.0 + testSeed))) * 0.2;
    float yTimeMultiplier = (0.5 - rand(vec2(float(i), 0.0 + testSeed))) * 0.2;

    float xRootPos = (0.75 - rand(vec2(float(i), 0.0 + testSeed))) * 2.0;
    float yRootPos = (0.75 - rand(vec2(float(i), 1.0 + testSeed))) * 2.0;

    float xPos = xRootPos + cos(time * xTimeMultiplier) * xOffsetMultiplier;
    float yPos = yRootPos + sin(time * yTimeMultiplier) * yOffsetMultiplier;

    lights[i] = Light(vec2(xPos, yPos), lightColor / 255.0, 1.0 + rand(vec2(float(i), 0.0)) * 2.0);

    vec3 rawLightColor = lights[i].color * max(lights[i].radius - distance(uv * vec2(canvasAspect, 1.0), lights[i].position * vec2(canvasAspect, 1.0)) * lights[i].radius, 0.0);

    color += rawLightColor;
  }

  color /= float(lights.length());


  color = min(color, vec3(1.0));

  /* color -= abs(BitangentNoise3D(vec3(vUv * 1000.0, 0.0) ).x) * 0.01; */

  gl_FragColor = vec4(color, 1.0);
}
`;

var vert
	= /* glsl */ `

precision highp float;
varying vec2 vUv;

void main() {
    vUv = uv;
    gl_Position = vec4( position / vec3(0.5), 1.0 );
}
`;

