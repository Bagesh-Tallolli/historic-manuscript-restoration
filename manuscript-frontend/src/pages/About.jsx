import React from 'react';
import { FaUniversity, FaLightbulb, FaCogs, FaHeart } from 'react-icons/fa';

const About = () => {
  const technologies = [
    { name: 'Python 3', purpose: 'Core programming language with AI/ML support' },
    { name: 'PyTorch', purpose: 'Deep learning framework for model training' },
    { name: 'Vision Transformer (ViT)', purpose: 'Image restoration for degraded manuscripts' },
    { name: 'Tesseract OCR', purpose: 'Sanskrit text extraction in Devanagari script' },
    { name: 'mBART', purpose: 'Multilingual neural machine translation' },
    { name: 'OpenCV', purpose: 'Image preprocessing (CLAHE & unsharp mask)' },
    { name: 'React', purpose: 'Modern UI framework' },
    { name: 'Streamlit', purpose: 'Backend web interface' },
    { name: 'Kaggle GPU', purpose: 'Training and processing environment' },
  ];

  const team = [
    { role: 'Project Lead', description: 'Academic research and coordination' },
    { role: 'AI/ML Engineer', description: 'OCR and translation models' },
    { role: 'Frontend Developer', description: 'React UI/UX implementation' },
    { role: 'Backend Developer', description: 'Python image processing' },
  ];

  return (
    <div className="min-h-screen bg-parchment-100 py-8 page-transition overflow-hidden max-w-full">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 overflow-hidden">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-heritage-600 mb-4">About This Project</h1>
          <p className="text-xl text-heritage-500 font-serif italic">
            Bridging Ancient Wisdom with Modern Technology
          </p>
        </div>

        {/* Mission */}
        <div className="card mb-8">
          <div className="flex items-start space-x-4">
            <FaLightbulb className="text-4xl text-saffron-500 mt-1 flex-shrink-0" />
            <div>
              <h2 className="text-2xl font-bold text-heritage-600 mb-3">Our Mission</h2>
              <p className="text-lg text-heritage-600 leading-relaxed mb-4">
                Sanskrit manuscripts hold thousands of years of knowledge in philosophy, science, mathematics,
                medicine, and literature. However, many of these invaluable texts remain inaccessible due to
                deterioration, poor preservation, or linguistic barriers.
              </p>
              <p className="text-lg text-heritage-600 leading-relaxed">
                This project aims to democratize access to ancient Indian heritage by leveraging cutting-edge
                AI technology for digital restoration, text extraction, and multi-language translation—making
                this wisdom accessible to scholars, researchers, and enthusiasts worldwide.
              </p>
            </div>
          </div>
        </div>

        {/* Academic Importance */}
        <div className="card mb-8">
          <div className="flex items-start space-x-4">
            <FaUniversity className="text-4xl text-saffron-500 mt-1 flex-shrink-0" />
            <div>
              <h2 className="text-2xl font-bold text-heritage-600 mb-3">Academic & Cultural Impact</h2>
              <ul className="space-y-3 text-heritage-600">
                <li className="flex items-start">
                  <span className="text-saffron-500 mr-2 text-xl">•</span>
                  <span><strong>Preservation:</strong> Digitally safeguard deteriorating manuscripts for future generations</span>
                </li>
                <li className="flex items-start">
                  <span className="text-saffron-500 mr-2 text-xl">•</span>
                  <span><strong>Accessibility:</strong> Enable global researchers to access rare texts remotely</span>
                </li>
                <li className="flex items-start">
                  <span className="text-saffron-500 mr-2 text-xl">•</span>
                  <span><strong>Translation:</strong> Break language barriers with accurate AI-powered translations</span>
                </li>
                <li className="flex items-start">
                  <span className="text-saffron-500 mr-2 text-xl">•</span>
                  <span><strong>Education:</strong> Facilitate academic research and education in Indology</span>
                </li>
                <li className="flex items-start">
                  <span className="text-saffron-500 mr-2 text-xl">•</span>
                  <span><strong>Heritage:</strong> Promote awareness of India's rich intellectual traditions</span>
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* Technology Stack */}
        <div className="card mb-8">
          <div className="flex items-start space-x-4">
            <FaCogs className="text-4xl text-saffron-500 mt-1 flex-shrink-0" />
            <div className="flex-1">
              <h2 className="text-2xl font-bold text-heritage-600 mb-4">Technology Stack</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {technologies.map((tech, index) => (
                  <div key={index} className="bg-parchment-50 p-4 rounded-lg border border-heritage-200">
                    <h3 className="font-bold text-heritage-600 mb-1">{tech.name}</h3>
                    <p className="text-sm text-heritage-500">{tech.purpose}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Technical Architecture */}
        <div className="card mb-8 bg-gradient-to-br from-blue-50 to-purple-50 border-2 border-blue-300">
          <div className="flex items-start space-x-4">
            <FaCogs className="text-4xl text-blue-600 mt-1 flex-shrink-0" />
            <div className="flex-1">
              <h2 className="text-2xl font-bold text-heritage-600 mb-4">Technical Architecture</h2>
              <div className="space-y-3 text-heritage-600">
                <div className="bg-white p-3 rounded-lg">
                  <strong className="text-saffron-600">Core Language:</strong> Python 3 with strong AI and image processing support
                </div>
                <div className="bg-white p-3 rounded-lg">
                  <strong className="text-saffron-600">Deep Learning Framework:</strong> PyTorch for model implementation and training
                </div>
                <div className="bg-white p-3 rounded-lg">
                  <strong className="text-saffron-600">Image Restoration:</strong> Vision Transformer (ViT) model for restoring degraded manuscripts
                </div>
                <div className="bg-white p-3 rounded-lg">
                  <strong className="text-saffron-600">OCR Engine:</strong> Tesseract OCR fine-tuned for Devanagari script extraction
                </div>
                <div className="bg-white p-3 rounded-lg">
                  <strong className="text-saffron-600">Translation:</strong> mBART pretrained multilingual neural machine translation model
                </div>
                <div className="bg-white p-3 rounded-lg">
                  <strong className="text-saffron-600">Training Environment:</strong> GPU-enabled Kaggle environment for efficient processing
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Team Roles */}
        <div className="card mb-8">
          <h2 className="text-2xl font-bold text-heritage-600 mb-6 text-center">Project Team</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {team.map((member, index) => (
              <div key={index} className="bg-saffron-50 p-5 rounded-lg border border-saffron-200 text-center">
                <h3 className="font-bold text-lg text-heritage-600 mb-2">{member.role}</h3>
                <p className="text-sm text-heritage-500">{member.description}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Pipeline Details */}
        <div className="card mb-8">
          <h2 className="text-2xl font-bold text-heritage-600 mb-4">Processing Pipeline</h2>
          <div className="space-y-4">
            <div className="border-l-4 border-saffron-400 pl-4">
              <h3 className="font-bold text-heritage-600">1. Image Upload & Preprocessing</h3>
              <p className="text-sm text-heritage-500">OpenCV-based enhancement using CLAHE and Unsharp Mask</p>
            </div>
            <div className="border-l-4 border-saffron-400 pl-4">
              <h3 className="font-bold text-heritage-600">2. ViT Model Restoration</h3>
              <p className="text-sm text-heritage-500">Vision Transformer restores degraded manuscript regions</p>
            </div>
            <div className="border-l-4 border-saffron-400 pl-4">
              <h3 className="font-bold text-heritage-600">3. Tesseract OCR Extraction</h3>
              <p className="text-sm text-heritage-500">Devanagari script recognition and Sanskrit text extraction</p>
            </div>
            <div className="border-l-4 border-saffron-400 pl-4">
              <h3 className="font-bold text-heritage-600">4. mBART Translation</h3>
              <p className="text-sm text-heritage-500">Neural machine translation to English, Hindi, and Kannada</p>
            </div>
          </div>
        </div>

        {/* Academic Use */}
        <div className="card bg-gradient-to-br from-saffron-50 to-parchment-100 border-2 border-saffron-300">
          <div className="flex items-start space-x-4">
            <FaHeart className="text-4xl text-red-500 mt-1 flex-shrink-0" />
            <div>
              <h2 className="text-2xl font-bold text-heritage-600 mb-3">Suitable For</h2>
              <ul className="space-y-2 text-heritage-600">
                <li className="flex items-center">
                  <span className="text-saffron-500 mr-2">✓</span>
                  Academic research institutions
                </li>
                <li className="flex items-center">
                  <span className="text-saffron-500 mr-2">✓</span>
                  Digital heritage libraries and archives
                </li>
                <li className="flex items-center">
                  <span className="text-saffron-500 mr-2">✓</span>
                  Sanskrit scholars and translators
                </li>
                <li className="flex items-center">
                  <span className="text-saffron-500 mr-2">✓</span>
                  Final-year engineering projects
                </li>
                <li className="flex items-center">
                  <span className="text-saffron-500 mr-2">✓</span>
                  Cultural heritage preservation initiatives
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* CTA */}
        <div className="text-center mt-8">
          <a href="/upload" className="btn-primary inline-block text-lg">
            Start Processing Manuscripts
          </a>
        </div>
      </div>
    </div>
  );
};

export default About;

