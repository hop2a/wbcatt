import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2 } from 'lucide-react';

const ImageAnalysisInterface = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedModel, setSelectedModel] = useState('resnet50');
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const models = [
    { value: 'resnet50', label: 'ResNet 50' },
    { value: 'vgg16', label: 'VGG 16' },
    { value: 'convnext_tiny', label: 'ConvNext Tiny' },
    { value: 'vit_b_16', label: 'Vision Transformer' }
  ];

  const attributeLabels = {
    cell_size: 'Taille de la cellule',
    cell_shape: 'Forme de la cellule',
    nucleus_shape: 'Forme du noyau',
    nuclear_cytoplasmic_ratio: 'Ratio nucléo-cytoplasmique',
    chromatin_density: 'Densité de la chromatine',
    cytoplasm_texture: 'Texture du cytoplasme',
    cytoplasm_colour: 'Couleur du cytoplasme',
    granule_type: 'Type de granules',
    granule_colour: 'Couleur des granules',
    granularity: 'Granularité'
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setError(null);
    }
  };

  const handleAnalysis = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('model', selectedModel);

    try {
      const response = await fetch('http://localhost:5000/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Erreur lors de l\'analyse');
      }

      const data = await response.json();
      setAnalysisResults(data.predictions);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto p-4">
      <Card>
        <CardHeader>
          <CardTitle>Analyse de Cellules Sanguines</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                Sélectionner une image
              </label>
              <Input
                type="file"
                accept="image/*"
                onChange={handleFileUpload}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Modèle d'analyse
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full p-2 border rounded"
              >
                {models.map((model) => (
                  <option key={model.value} value={model.value}>
                    {model.label}
                  </option>
                ))}
              </select>
            </div>

            {error && (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <Button 
              onClick={handleAnalysis}
              disabled={!selectedFile || isLoading}
              className="w-full"
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Analyse en cours...
                </>
              ) : (
                'Analyser l\'image'
              )}
            </Button>

            {analysisResults && (
              <div className="mt-4">
                <h3 className="text-lg font-medium mb-2">Résultats</h3>
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(analysisResults).map(([key, value]) => (
                    <div key={key} className="p-2 bg-gray-100 rounded">
                      <span className="font-medium">
                        {attributeLabels[key] || key}:
                      </span>{' '}
                      <div className="text-sm mt-1">
                        <div>Classe: {value.class}</div>
                        <div>Confiance: {(value.confidence * 100).toFixed(2)}%</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ImageAnalysisInterface;