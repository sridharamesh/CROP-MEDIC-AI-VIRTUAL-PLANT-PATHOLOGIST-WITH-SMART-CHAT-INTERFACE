# Plant Assistant 🌱

Plant Assistant is an AI-powered application that helps identify plant diseases and provides treatment recommendations.

## Features

- **Plant Disease Detection**: Upload images of plant leaves to identify diseases
- **Treatment Recommendations**: Get actionable advice for treating identified plant issues
- **User-Friendly Interface**: Simple web interface for easy access and use

## Installation

### Prerequisites

- Python 3.8 or higher
- Git
- Conda (for environment management)

### Step 1: Clone the Repository

```bash
git clone https://github.com/sridharamesh/Plant_assistant.git
cd Plant_assistant
```

### Step 2: Create and Activate Conda Environment

```bash
# Create a new conda environment
conda create -n plant_env python=3.8

# Activate the environment
conda activate plant_env
```

### Step 3: Install Required Dependencies

```bash
pip install -r requirements.txt

install ollama from ollama.ai website
pull mistral and mxbai-embed-large
ollama pull mistral
ollama pull mxbai-embed-large 
```

### Step 4: Download the Pre-trained Model

```bash
# Download the model file from Google Drive
gdown https://drive.google.com/file/d/1nLD00rpwNn6PfljMuUu2v7AxxWenRwqU/view?usp=drive_link
```

## Usage

### Starting the Application

```bash
python app.py
```

After running this command, the web interface will be available at `http://localhost:5000` (or another port if specified in the application).

### Using the Interface

1. Navigate to the web interface in your browser
2. Upload an image of a plant leaf
3. Submit for analysis
4. View the disease identification results and treatment recommendations

## Technical Details

The Plant Assistant uses deep learning models to identify plant diseases from images. The application is built using:

- **Backend**: Python with Flask
- **Machine Learning**: TensorFlow/PyTorch (specify which one your project uses)
- **Frontend**: HTML, CSS, JavaScript

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- List any datasets used for training
- Credit any research papers or methodologies that informed your approach
- Acknowledge contributors or inspirations

---

Developed by [Sridhar Ramesh](https://github.com/sridharamesh)
