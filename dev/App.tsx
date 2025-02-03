import React from 'react'
import { Button, ShaderTest } from 'your-lib'

const App = () => {
  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <h1 className="text-2xl font-bold mb-8">Component Demo</h1>
      
      <div className="space-y-8">
        <div>
          <h2 className="text-lg font-semibold mb-4">Shader Test</h2>
          <div className="w-full max-w-2xl h-[512px]">
            <ShaderTest />
          </div>
        </div>
        
        <div>
          <h2 className="text-lg font-semibold mb-4">Buttons</h2>
          <div className="space-x-4">
            <Button variant="primary">Primary Button</Button>
            <Button variant="secondary">Secondary Button</Button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App