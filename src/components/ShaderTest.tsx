// ShaderTest.tsx
import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
// Import your shader files
import vertexShader from '../shaders/test.vert';
import fragmentShader from '../shaders/test.frag';

export const ShaderTest = () => {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    const scene = new THREE.Scene();
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
    const renderer = new THREE.WebGLRenderer({ canvas: canvasRef.current });
    renderer.setSize(512, 512);
    camera.position.z = 1;

    const geometry = new THREE.PlaneGeometry(2, 2);
    
    const material = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 }
      },
      vertexShader,
      fragmentShader
    });

    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);

    const animate = () => {
      material.uniforms.time.value = performance.now() / 1000;
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    };

    animate();

    return () => {
      geometry.dispose();
      material.dispose();
      renderer.dispose();
    };
  }, []);

  return (
    <div className="w-full h-full flex items-center justify-center bg-gray-900">
      <canvas
        ref={canvasRef}
        className="w-96 h-96 rounded-lg shadow-xl"
      />
    </div>
  );
};

export default ShaderTest;