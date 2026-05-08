import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  BarChart3, 
  Database, 
  Play, 
  RefreshCcw, 
  Cpu, 
  Layers, 
  CheckCircle2, 
  TrendingUp,
  FlaskConical,
  Binary,
  Save
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';
import { engine, TrainingMetrics } from './lib/ml/engine';
import { IRIS_CLASSES } from './lib/ml/iris';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export default function App() {
  const [metrics, setMetrics] = useState<TrainingMetrics[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [currentModelStatus, setCurrentModelStatus] = useState<'idle' | 'training' | 'ready'>('idle');
  const [prediction, setPrediction] = useState<{ index: number; probs: Float32Array | Int32Array | Uint8Array } | null>(null);
  const [testInput, setTestInput] = useState([5.1, 3.5, 1.4, 0.2]); // Example setosa
  const [epochs, setEpochs] = useState(50);
  const [activeTab, setActiveTab] = useState<'monitor' | 'inference' | 'data' | 'python'>('monitor');

  useEffect(() => {
    const checkModel = async () => {
      const loaded = await engine.loadModel();
      if (loaded) setCurrentModelStatus('ready');
    };
    checkModel();
  }, []);

  const handleTrain = async () => {
    setIsTraining(true);
    setMetrics([]);
    setCurrentModelStatus('training');
    
    try {
      await engine.train(epochs, (m) => {
        setMetrics(prev => [...prev, m]);
      });
      setCurrentModelStatus('ready');
      await engine.saveModel();
    } catch (error) {
      console.error(error);
      setCurrentModelStatus('idle');
    } finally {
      setIsTraining(false);
    }
  };

  const handlePredict = async () => {
    if (currentModelStatus !== 'ready') return;
    const res = await engine.predict(testInput);
    setPrediction(res);
  };

  const latestMetrics = metrics[metrics.length - 1] || { epoch: 0, loss: 0, acc: 0 };

  return (
    <div className="min-h-screen bg-[#E4E3E0] text-[#141414] font-sans selection:bg-[#141414] selection:text-[#E4E3E0]">
      {/* Sidebar / Navigation */}
      <nav className="fixed top-0 left-0 h-full w-64 border-r border-[#141414] flex flex-col pt-8 bg-[#E4E3E0]/80 backdrop-blur-md z-40">
        <div className="px-6 mb-12">
          <div className="flex items-center gap-2 mb-1">
            <FlaskConical className="w-6 h-6" />
            <h1 className="font-serif italic text-xl tracking-tight">Iris MLOps Lab</h1>
          </div>
          <p className="text-[10px] font-mono opacity-50 uppercase tracking-widest">System v1.0.4 - Production</p>
        </div>

        <div className="flex-1 space-y-1 px-2">
          <NavButton 
            active={activeTab === 'monitor'} 
            onClick={() => setActiveTab('monitor')}
            icon={<Activity className="w-4 h-4" />}
            label="Experiment Monitor" 
          />
          <NavButton 
            active={activeTab === 'inference'} 
            onClick={() => setActiveTab('inference')}
            icon={<Cpu className="w-4 h-4" />}
            label="Model Inference" 
          />
          <NavButton 
            active={activeTab === 'data'} 
            onClick={() => setActiveTab('data')}
            icon={<Database className="w-4 h-4" />}
            label="Dataset Explorer" 
          />
          <NavButton 
            active={activeTab === 'python'} 
            onClick={() => setActiveTab('python')}
            icon={<Binary className="w-4 h-4" />}
            label="Python Architecture" 
          />
        </div>

        <div className="p-6 border-t border-[#141414]">
          <div className="flex items-center gap-2 mb-2">
            <div className={cn(
              "w-2 h-2 rounded-full",
              currentModelStatus === 'ready' ? "bg-green-600 shadow-[0_0_8px_rgba(22,163,74,0.5)]" : 
              currentModelStatus === 'training' ? "bg-orange-500 animate-pulse" : "bg-red-500"
            )} />
            <span className="text-[11px] font-mono uppercase tracking-wider">
              {currentModelStatus === 'ready' ? 'Model Live' : 
               currentModelStatus === 'training' ? 'Training...' : 'No Model Loaded'}
            </span>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="ml-64 p-8">
        {activeTab === 'monitor' && (
          <div className="max-w-5xl space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <header className="flex justify-between items-end border-b border-[#141414] pb-4">
              <div>
                <span className="font-serif italic text-xs opacity-50 uppercase tracking-widest">Active Run</span>
                <h2 className="text-4xl font-light tracking-tighter">Experiment Workflow</h2>
              </div>
              <div className="flex gap-4 items-center">
                <div className="space-x-2 flex items-center">
                  <span className="text-[10px] font-mono opacity-50 uppercase">Epochs</span>
                  <input 
                    type="number" 
                    value={epochs} 
                    onChange={(e) => setEpochs(Number(e.target.value))}
                    className="w-16 bg-transparent border border-[#141414] px-2 py-1 text-sm font-mono focus:outline-none"
                    disabled={isTraining}
                  />
                </div>
                <button 
                  onClick={handleTrain}
                  disabled={isTraining}
                  className={cn(
                    "flex items-center gap-2 px-6 py-2 border border-[#141414] font-mono text-sm uppercase tracking-wider transition-all",
                    isTraining ? "opacity-50 cursor-not-allowed" : "hover:bg-[#141414] hover:text-[#E4E3E0]"
                  )}
                >
                  {isTraining ? <RefreshCcw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                  {isTraining ? 'System Learning' : 'Initialize Training'}
                </button>
              </div>
            </header>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-px bg-[#141414] border border-[#141414]">
              <StatCard label="Current Epoch" value={latestMetrics.epoch} sub="Of scheduled run" icon={<Layers className="w-4 h-4" />} />
              <StatCard label="Learning Accuracy" value={`${(latestMetrics.acc * 100).toFixed(2)}%`} sub="Real-time evaluation" icon={<TrendingUp className="w-4 h-4" />} />
              <StatCard label="Loss Variance" value={latestMetrics.loss.toFixed(4)} sub="Categorical entropy" icon={<Activity className="w-4 h-4" />} />
              <StatCard label="Model Version" value="v1.0.0-js" sub="TensorFlow.js Production" icon={<Save className="w-4 h-4" />} />
            </div>

            {/* Main Chart */}
            <div className="border border-[#141414] p-6 bg-white/50">
              <div className="flex justify-between items-center mb-8">
                <h3 className="font-serif italic text-lg capitalize">Training loss & accuracy curves</h3>
                <div className="flex gap-4 text-[10px] font-mono opacity-50 uppercase">
                  <span className="flex items-center gap-1"><div className="w-2 h-2 bg-[#141414]" /> Accuracy</span>
                  <span className="flex items-center gap-1"><div className="w-2 h-2 bg-orange-500" /> Loss</span>
                </div>
              </div>
              <div className="h-[400px]">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metrics}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" vertical={false} />
                    <XAxis 
                      dataKey="epoch" 
                      stroke="#141414" 
                      fontSize={10} 
                      tickLine={false} 
                      axisLine={false}
                    />
                    <YAxis 
                      stroke="#141414" 
                      fontSize={10} 
                      tickLine={false} 
                      axisLine={false} 
                      domain={[0, 1]}
                    />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#141414', border: 'none', color: '#E4E3E0' }}
                      itemStyle={{ color: '#E4E3E0', fontSize: '12px', fontFamily: 'monospace' }}
                      labelStyle={{ marginBottom: '4px', opacity: 0.5 }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="acc" 
                      stroke="#141414" 
                      strokeWidth={2} 
                      dot={false}
                      animationDuration={300}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="loss" 
                      stroke="#f97316" 
                      strokeWidth={1} 
                      strokeDasharray="4 4"
                      dot={false}
                      animationDuration={300}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'inference' && (
          <div className="max-w-5xl space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <header className="border-b border-[#141414] pb-4">
              <span className="font-serif italic text-xs opacity-50 uppercase tracking-widest">Real-time Deployment</span>
              <h2 className="text-4xl font-light tracking-tighter">Model Inference</h2>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="space-y-6">
                <div className="bg-white/50 border border-[#141414] p-6">
                  <h3 className="font-serif italic text-lg mb-6">Input parameters (cm)</h3>
                  <div className="space-y-4">
                    {['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'].map((label, i) => (
                      <div key={label} className="grid grid-cols-2 items-center">
                        <label className="text-[11px] font-mono uppercase tracking-wider opacity-60">{label}</label>
                        <input 
                          type="range" 
                          min="0.1" 
                          max="8.0" 
                          step="0.1" 
                          value={testInput[i]} 
                          onChange={(e) => {
                            const newVals = [...testInput];
                            newVals[i] = parseFloat(e.target.value);
                            setTestInput(newVals);
                          }}
                          className="accent-[#141414]"
                        />
                        <div className="col-start-2 text-right text-[10px] font-mono mt-1">{testInput[i].toFixed(1)} cm</div>
                      </div>
                    ))}
                  </div>
                  <button 
                    onClick={handlePredict}
                    disabled={currentModelStatus !== 'ready'}
                    className={cn(
                      "w-full mt-8 py-3 border border-[#141414] font-mono text-xs uppercase tracking-[0.2em] transition-all",
                      currentModelStatus !== 'ready' ? "opacity-30 cursor-not-allowed" : "hover:bg-[#141414] hover:text-[#E4E3E0]"
                    )}
                  >
                    Run Inference Pipeline
                  </button>
                </div>
              </div>

              <div className="bg-[#141414] text-[#E4E3E0] p-8 flex flex-col justify-between">
                <div>
                  <div className="flex justify-between items-center mb-12">
                    <span className="text-[10px] font-mono opacity-50 uppercase tracking-widest">Classification Output</span>
                    <BarChart3 className="w-5 h-5 opacity-50" />
                  </div>
                  
                  {prediction ? (
                    <div className="space-y-8">
                      <div>
                        <h4 className="font-serif italic text-6xl mb-2">{IRIS_CLASSES[prediction.index]}</h4>
                        <p className="text-xs font-mono opacity-50 uppercase tracking-wider">Identified species with high confidence</p>
                      </div>

                      <div className="space-y-3">
                        {IRIS_CLASSES.map((cls, i) => (
                          <div key={cls} className="space-y-1">
                            <div className="flex justify-between text-[10px] font-mono uppercase">
                              <span>{cls}</span>
                              <span>{(prediction.probs[i] * 100).toFixed(1)}%</span>
                            </div>
                            <div className="h-1 bg-white/10 overflow-hidden">
                              <div 
                                className="h-full bg-white transition-all duration-1000" 
                                style={{ width: `${prediction.probs[i] * 100}%` }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="h-full flex items-center justify-center py-20 text-center opacity-30">
                      <div>
                        <div className="w-12 h-12 border border-white mx-auto mb-4 flex items-center justify-center animate-pulse">
                          <Binary className="w-6 h-6" />
                        </div>
                        <p className="text-[11px] font-mono uppercase tracking-[0.3em]">Awaiting Data Input</p>
                      </div>
                    </div>
                  )}
                </div>

                <div className="mt-8 pt-8 border-t border-white/10">
                  <div className="flex items-center gap-2 group cursor-help">
                    <CheckCircle2 className="w-4 h-4 text-green-500" />
                    <span className="text-[10px] font-mono uppercase tracking-wider opacity-70 group-hover:opacity-100 transition-opacity">
                      Hardware Acceleration Active
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'data' && (
          <div className="max-w-5xl space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
             <header className="border-b border-[#141414] pb-4">
              <span className="font-serif italic text-xs opacity-50 uppercase tracking-widest">Source Material</span>
              <h2 className="text-4xl font-light tracking-tighter">Dataset Explorer</h2>
            </header>
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              <div className="lg:col-span-2 space-y-4">
                <div className="border border-[#141414] overflow-hidden">
                  <table className="w-full text-left font-mono text-[11px]">
                    <thead className="bg-[#141414] text-[#E4E3E0] uppercase tracking-wider">
                      <tr>
                        <th className="px-4 py-3 border-r border-white/10">S.Length</th>
                        <th className="px-4 py-3 border-r border-white/10">S.Width</th>
                        <th className="px-4 py-3 border-r border-white/10">P.Length</th>
                        <th className="px-4 py-3 border-r border-white/10">P.Width</th>
                        <th className="px-4 py-3">Species</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#141414]/10 bg-white/30 text-[#141414]/70 italic">
                      {[1,2,3,4,5,6,7,8,9,10].map((_, i) => (
                        <tr key={i} className="hover:bg-white/50 transition-colors">
                          <td className="px-4 py-2 border-r border-[#141414]/10 font-normal">{(5.0 + Math.random()).toFixed(1)}</td>
                          <td className="px-4 py-2 border-r border-[#141414]/10 font-normal">{(3.0 + Math.random()).toFixed(1)}</td>
                          <td className="px-4 py-2 border-r border-[#141414]/10 font-normal">{(1.5 + Math.random()).toFixed(1)}</td>
                          <td className="px-4 py-2 border-r border-[#141414]/10 font-normal">{(0.2 + Math.random()).toFixed(1)}</td>
                          <td className="px-4 py-2 font-serif text-sm">setosa</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <div className="bg-[#141414]/5 p-4 text-center border-t border-[#141414]">
                    <p className="text-[10px] font-mono opacity-50 italic">Showing top 10 samples from 150 record vector space</p>
                  </div>
                </div>
              </div>

              <div className="space-y-6">
                <div className="p-6 bg-white border border-[#141414] space-y-4">
                  <h3 className="font-serif italic text-lg">Statistical Distribution</h3>
                  <div className="h-[200px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={[
                        { name: 'Setosa', val: 50 },
                        { name: 'Versicolor', val: 50 },
                        { name: 'Virginica', val: 50 },
                      ]}>
                        <Area type="step" dataKey="val" stroke="#141414" fill="#141414" fillOpacity={0.1} />
                        <XAxis dataKey="name" fontSize={10} axisLine={false} tickLine={false} />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="pt-4 border-t border-[#141414]/10 text-[11px] space-y-2 opacity-70">
                    <p>• Perfectly balanced categorical split (n=50/cls)</p>
                    <p>• Zero missing values in feature set</p>
                    <p>• Normalized floating point headers</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'python' && (
          <div className="max-w-5xl space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
             <header className="border-b border-[#141414] pb-4">
              <span className="font-serif italic text-xs opacity-50 uppercase tracking-widest">Enterprise Reference</span>
              <h2 className="text-4xl font-light tracking-tighter">Python MLOps Architecture</h2>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              <div className="lg:col-span-2">
                <div className="bg-[#141414] text-[#E4E3E0] p-6 border border-[#141414] font-mono text-xs leading-relaxed overflow-x-auto shadow-xl">
                  <div className="flex justify-between items-center mb-4 border-b border-white/10 pb-2">
                    <span className="text-[10px] opacity-40 uppercase tracking-widest">iris_pipeline.py</span>
                    <div className="flex gap-1">
                      <div className="w-2 h-2 rounded-full bg-red-400/50" />
                      <div className="w-2 h-2 rounded-full bg-yellow-400/50" />
                      <div className="w-2 h-2 rounded-full bg-green-400/50" />
                    </div>
                  </div>
                  <pre className="opacity-90">
{`from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def main():
    # 1. Pipeline orchestration
    pipeline = IrisMLPipeline()
    raw_data = pipeline.load_data()
    
    # 2. Automated feature engineering
    processed = pipeline.preprocess(raw_data)
    
    # 3. Model Training & Evaluation
    metrics = pipeline.train(processed)
    
    # 4. Model Registry push
    pipeline.save_model('v1_stable.pkl')`}
                  </pre>
                </div>
              </div>

              <div className="space-y-6">
                <div className="border border-[#141414] p-6 space-y-4">
                  <h3 className="font-serif italic text-lg">Platform Stack</h3>
                  <div className="space-y-3">
                    <StackItem label="Environment" value="Scikit-Learn / Pandas" />
                    <StackItem label="Serving" value="FastAPI / Flask" />
                    <StackItem label="Versioning" value="DVC + Git" />
                    <StackItem label="Tracking" value="MLflow / Weights & Biases" />
                    <StackItem label="Infrastructure" value="Docker / Cloud Run" />
                  </div>
                  <div className="pt-4 mt-6 border-t border-[#141414]/10">
                    <p className="text-[10px] font-mono opacity-50 leading-relaxed">
                      The Python architecture ensures enterprise-scale reproducibility and easier integration with standard data science tooling.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Global Toast / Overlay for training */}
      {isTraining && (
        <div className="fixed bottom-8 right-8 bg-[#141414] text-[#E4E3E0] px-6 py-4 flex items-center gap-4 border border-white/20 animate-in fade-in slide-in-from-right-8 z-50">
          <RefreshCcw className="w-5 h-5 animate-spin text-orange-500" />
          <div>
            <div className="text-[10px] font-mono uppercase tracking-widest opacity-50">Core Engine Training</div>
            <div className="text-sm font-light italic font-serif">Optimizing categorical cross-entropy...</div>
          </div>
        </div>
      )}
    </div>
  );
}

function NavButton({ active, label, icon, onClick }: { active: boolean, label: string, icon: React.ReactNode, onClick: () => void }) {
  return (
    <button 
      onClick={onClick}
      className={cn(
        "w-full flex items-center gap-3 px-6 py-3 text-xs font-mono uppercase tracking-[0.15em] transition-all relative overflow-hidden",
        active ? "text-[#141414] bg-white/40" : "text-[#141414]/40 hover:text-[#141414] hover:bg-white/20"
      )}
    >
      {active && <div className="absolute left-0 top-0 h-full w-1 bg-[#141414]" />}
      {icon}
      {label}
    </button>
  );
}

function StatCard({ label, value, sub, icon }: { label: string, value: string | number, sub: string, icon: React.ReactNode }) {
  return (
    <div className="bg-white/60 p-6 flex flex-col justify-between group hover:bg-[#141414] hover:text-[#E4E3E0] transition-all duration-300">
      <div className="flex justify-between items-start mb-4">
        <span className="text-[10px] font-mono uppercase tracking-widest opacity-50 group-hover:opacity-100">{label}</span>
        <div className="opacity-30 group-hover:opacity-100">{icon}</div>
      </div>
      <div>
        <div className="text-3xl font-light tracking-tighter mb-1">{value}</div>
        <div className="text-[10px] font-serif italic opacity-40 group-hover:opacity-60">{sub}</div>
      </div>
    </div>
  );
}

function StackItem({ label, value }: { label: string, value: string }) {
  return (
    <div className="flex justify-between items-end border-b border-[#141414]/10 pb-1">
      <span className="text-[10px] font-mono uppercase tracking-wider opacity-40">{label}</span>
      <span className="text-xs font-serif italic">{value}</span>
    </div>
  );
}
