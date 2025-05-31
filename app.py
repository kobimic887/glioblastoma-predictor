from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import joblib
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

class DrugSensitivityPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        
    def smiles_to_features(self, smiles):
        """Convert SMILES string to molecular descriptors"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Calculate molecular descriptors
            features = {
                'MolWt': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'NumHDonors': Descriptors.NumHDonors(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                'TPSA': Descriptors.TPSA(mol),
                'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
                'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
                'RingCount': Descriptors.RingCount(mol),
                'FractionCsp3': Descriptors.FractionCsp3(mol),
                'NumHeteroatoms': Descriptors.NumHeteroatoms(mol),
                'BertzCT': Descriptors.BertzCT(mol),
                'Chi0v': Descriptors.Chi0v(mol),
                'Chi1v': Descriptors.Chi1v(mol),
                'Chi2v': Descriptors.Chi2v(mol),
                'Chi3v': Descriptors.Chi3v(mol),
                'Chi4v': Descriptors.Chi4v(mol),
                'Kappa1': Descriptors.Kappa1(mol),
                'Kappa2': Descriptors.Kappa2(mol),
                'Kappa3': Descriptors.Kappa3(mol)
            }
            
            return np.array(list(features.values())).reshape(1, -1)
            
        except Exception as e:
            logging.error(f"Error processing SMILES {smiles}: {str(e)}")
            return None
    
    def train_mock_model(self):
        """Train a mock model with synthetic data for demonstration"""
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Mock molecular descriptors
        X = np.random.randn(n_samples, 21)  # 21 features as defined above
        # Mock sensitivity scores (IC50 values, lower = more sensitive)
        y = np.random.lognormal(mean=1, sigma=1, size=n_samples)
        
        # Train scaler and model
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        
        self.feature_names = [
            'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA',
            'NumRotatableBonds', 'NumAromaticRings', 'NumSaturatedRings',
            'NumAliphaticRings', 'RingCount', 'FractionCsp3', 'NumHeteroatoms',
            'BertzCT', 'Chi0v', 'Chi1v', 'Chi2v', 'Chi3v', 'Chi4v',
            'Kappa1', 'Kappa2', 'Kappa3'
        ]
        
        self.is_trained = True
        logging.info("Mock model trained successfully")
    
    def predict_sensitivity(self, smiles):
        """Predict drug sensitivity for given SMILES"""
        if not self.is_trained:
            return None, "Model not trained"
        
        features = self.smiles_to_features(smiles)
        if features is None:
            return None, "Invalid SMILES string"
        
        try:
            features_scaled = self.scaler.transform(features)
            prediction = self.model.predict(features_scaled)[0]
            
            # Convert to sensitivity score (lower IC50 = higher sensitivity)
            sensitivity_score = 1 / (1 + prediction)
            
            return {
                'ic50_prediction': float(prediction),
                'sensitivity_score': float(sensitivity_score),
                'sensitivity_category': self._categorize_sensitivity(sensitivity_score)
            }, None
            
        except Exception as e:
            return None, f"Prediction error: {str(e)}"
    
    def _categorize_sensitivity(self, score):
        """Categorize sensitivity based on score"""
        if score > 0.7:
            return "High Sensitivity"
        elif score > 0.4:
            return "Moderate Sensitivity"
        else:
            return "Low Sensitivity"

# Initialize predictor
predictor = DrugSensitivityPredictor()
predictor.train_mock_model()

@app.route('/', methods=['GET'])
def home():
    """API documentation endpoint"""
    return jsonify({
        "service": "Glioblastoma Drug Sensitivity Predictor",
        "version": "1.0.0",
        "description": "Predict drug sensitivity in glioblastoma using SMILES notation",
        "endpoints": {
            "/predict": {
                "method": "POST",
                "description": "Predict drug sensitivity for a given SMILES string",
                "parameters": {
                    "smiles": "SMILES notation of the drug molecule"
                }
            },
            "/batch_predict": {
                "method": "POST", 
                "description": "Predict drug sensitivity for multiple SMILES",
                "parameters": {
                    "smiles_list": "List of SMILES strings"
                }
            },
            "/health": {
                "method": "GET",
                "description": "Health check endpoint"
            }
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_trained": predictor.is_trained,
        "timestamp": pd.Timestamp.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict drug sensitivity for a single SMILES string"""
    try:
        data = request.get_json()
        
        if not data or 'smiles' not in data:
            return jsonify({
                "error": "Missing 'smiles' parameter in request body"
            }), 400
        
        smiles = data['smiles']
        
        if not isinstance(smiles, str) or not smiles.strip():
            return jsonify({
                "error": "SMILES must be a non-empty string"
            }), 400
        
        prediction, error = predictor.predict_sensitivity(smiles)
        
        if error:
            return jsonify({"error": error}), 400
        
        return jsonify({
            "smiles": smiles,
            "prediction": prediction,
            "status": "success"
        })
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict drug sensitivity for multiple SMILES strings"""
    try:
        data = request.get_json()
        
        if not data or 'smiles_list' not in data:
            return jsonify({
                "error": "Missing 'smiles_list' parameter in request body"
            }), 400
        
        smiles_list = data['smiles_list']
        
        if not isinstance(smiles_list, list):
            return jsonify({
                "error": "smiles_list must be a list"
            }), 400
        
        if len(smiles_list) > 100:  # Limit batch size
            return jsonify({
                "error": "Batch size limited to 100 SMILES"
            }), 400
        
        results = []
        
        for i, smiles in enumerate(smiles_list):
            if not isinstance(smiles, str) or not smiles.strip():
                results.append({
                    "smiles": smiles,
                    "error": "Invalid SMILES string",
                    "index": i
                })
                continue
            
            prediction, error = predictor.predict_sensitivity(smiles)
            
            if error:
                results.append({
                    "smiles": smiles,
                    "error": error,
                    "index": i
                })
            else:
                results.append({
                    "smiles": smiles,
                    "prediction": prediction,
                    "index": i,
                    "status": "success"
                })
        
        return jsonify({
            "results": results,
            "total_processed": len(smiles_list)
        })
        
    except Exception as e:
        logging.error(f"Batch prediction error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)