import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Play, Pause, RotateCcw } from "lucide-react";

const NeuralODEVisualizer = () => {
  const [trajectory, setTrajectory] = useState([]);
  const [currentTimeIdx, setCurrentTimeIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState(null);

  const fetchTrajectory = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8000/simulate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          n_cells: 100,
          time_points: Array.from({length: 100}, (_, i) => i * 0.1),
          initial_state: Array.from({length: 100}, () => [
            Math.random() * 2 - 1,
            Math.random() * 2 - 1,
            Math.random() * 2 - 1
          ])
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch trajectory');
      }
      
      const data = await response.json();
      setTrajectory(data.trajectory);
      setError(null);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching trajectory:', err);
    }
  }, []);

  useEffect(() => {
    fetchTrajectory();
  }, [fetchTrajectory]);

  useEffect(() => {
    let animationFrame;
    if (isPlaying && trajectory.length > 0) {
      const animate = () => {
        setCurrentTimeIdx(prev => (prev + 1) % trajectory.length);
        animationFrame = requestAnimationFrame(animate);
      };
      animationFrame = requestAnimationFrame(animate);
    }
    return () => cancelAnimationFrame(animationFrame);
  }, [isPlaying, trajectory.length]);

  const resetSimulation = () => {
    setCurrentTimeIdx(0);
    setIsPlaying(false);
    fetchTrajectory();
  };

  // Project 3D points to 2D with perspective
  const project = (x, y, z) => {
    const perspective = 500;
    const scaleFactor = perspective / (perspective + z);
    return {
      x: 400 + x * 50 * scaleFactor,
      y: 300 + y * 50 * scaleFactor,
      size: 4 * scaleFactor
    };
  };

  return (
    <Card className="w-full max-w-4xl">
      <CardHeader>
        <CardTitle>Embryo Development Simulation</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-col items-center gap-4">
          {error ? (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          ) : (
            <>
              <svg 
                width="800" 
                height="600" 
                className="border border-gray-200 rounded-lg bg-gray-50"
              >
                {trajectory[currentTimeIdx]?.map((cell, i) => {
                  const projected = project(cell[0], cell[1], cell[2]);
                  return (
                    <circle
                      key={i}
                      cx={projected.x}
                      cy={projected.y}
                      r={projected.size}
                      fill="blue"
                      opacity={0.6}
                    />
                  );
                })}
              </svg>
              
              <div className="flex gap-4 items-center">
                <button
                  className="p-2 rounded-full hover:bg-gray-100"
                  onClick={() => setIsPlaying(!isPlaying)}
                >
                  {isPlaying ? <Pause size={24} /> : <Play size={24} />}
                </button>
                <button
                  className="p-2 rounded-full hover:bg-gray-100"
                  onClick={resetSimulation}
                >
                  <RotateCcw size={24} />
                </button>
                <div className="text-sm text-gray-600">
                  Time step: {currentTimeIdx}
                </div>
              </div>
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default NeuralODEVisualizer;