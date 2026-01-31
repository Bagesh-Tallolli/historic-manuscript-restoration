import React, { useCallback, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { FaUpload, FaImage, FaCheckCircle, FaArrowRight } from 'react-icons/fa';
import { toast } from 'react-toastify';
import { useManuscript } from '../contexts/ManuscriptContext';
import StepProgressBar from '../components/StepProgressBar';

const Upload = () => {
  const navigate = useNavigate();
  const { setUploadedImage, setMetadata, setCurrentStep } = useManuscript();
  const [preview, setPreview] = useState(null);
  const [fileName, setFileName] = useState('');
  const [script, setScript] = useState('Devanagari');
  const [language, setLanguage] = useState('Sanskrit');

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      // Validate file type
      if (!['image/png', 'image/jpeg', 'image/jpg'].includes(file.type)) {
        toast.error('Please upload PNG, JPG, or JPEG files only');
        return;
      }

      // Validate file size (max 10MB)
      if (file.size > 10 * 1024 * 1024) {
        toast.error('File size must be less than 10MB');
        return;
      }

      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target.result);
        setUploadedImage(e.target.result);
        setFileName(file.name);
      };
      reader.readAsDataURL(file);
      toast.success('Image uploaded successfully!');
    }
  }, [setUploadedImage]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/png': ['.png'],
      'image/jpeg': ['.jpg', '.jpeg'],
    },
    maxFiles: 1,
  });

  const handleProceed = () => {
    if (!preview) {
      toast.error('Please upload an image first');
      return;
    }

    setMetadata({
      fileName,
      uploadDate: new Date(),
      script,
      language,
    });
    setCurrentStep(1);
    navigate('/restore');
  };

  return (
    <div className="min-h-screen bg-parchment-100 py-8 page-transition overflow-hidden max-w-full">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 overflow-hidden">
        <StepProgressBar />

        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-heritage-600 mb-3">Upload Manuscript</h1>
          <p className="text-lg text-heritage-400">
            Step 1 of 4: Select your Sanskrit manuscript image to begin the digitization process
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Area */}
          <div className="card">
            <h2 className="text-2xl font-bold text-heritage-600 mb-6 flex items-center">
              <FaUpload className="mr-3 text-saffron-500" />
              Upload Image
            </h2>

            <div
              {...getRootProps()}
              className={`border-3 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all duration-300 ${
                isDragActive
                  ? 'border-saffron-500 bg-saffron-50'
                  : 'border-heritage-300 bg-parchment-50 hover:border-saffron-400 hover:bg-saffron-50'
              }`}
            >
              <input {...getInputProps()} />
              <FaImage className="text-6xl text-heritage-300 mx-auto mb-4" />
              {isDragActive ? (
                <p className="text-lg text-heritage-600 font-medium">Drop the image here...</p>
              ) : (
                <>
                  <p className="text-lg text-heritage-600 font-medium mb-2">
                    Drag & drop your manuscript here
                  </p>
                  <p className="text-sm text-heritage-400 mb-4">or click to browse</p>
                  <button className="btn-primary inline-block">
                    Choose File
                  </button>
                  <p className="text-xs text-heritage-400 mt-4">
                    Supported: PNG, JPG, JPEG (Max 10MB)
                  </p>
                </>
              )}
            </div>

            {fileName && (
              <div className="mt-4 flex items-center text-green-600 bg-green-50 p-3 rounded-lg">
                <FaCheckCircle className="mr-2" />
                <span className="font-medium">{fileName}</span>
              </div>
            )}
          </div>

          {/* Metadata & Preview */}
          <div className="space-y-6">
            {/* Metadata Form */}
            <div className="card">
              <h2 className="text-2xl font-bold text-heritage-600 mb-6">Manuscript Details</h2>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-semibold text-heritage-600 mb-2">
                    Script Type
                  </label>
                  <select
                    value={script}
                    onChange={(e) => setScript(e.target.value)}
                    className="w-full px-4 py-3 border-2 border-heritage-200 rounded-lg focus:outline-none focus:border-saffron-400 bg-parchment-50"
                  >
                    <option value="Devanagari">Devanagari</option>
                    <option value="Telugu">Telugu</option>
                    <option value="Kannada">Kannada</option>
                    <option value="Tamil">Tamil</option>
                    <option value="Malayalam">Malayalam</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-semibold text-heritage-600 mb-2">
                    Language
                  </label>
                  <select
                    value={language}
                    onChange={(e) => setLanguage(e.target.value)}
                    className="w-full px-4 py-3 border-2 border-heritage-200 rounded-lg focus:outline-none focus:border-saffron-400 bg-parchment-50"
                  >
                    <option value="Sanskrit">Sanskrit</option>
                    <option value="Pali">Pali</option>
                    <option value="Prakrit">Prakrit</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Image Preview */}
            {preview && (
              <div className="card">
                <h3 className="text-xl font-bold text-heritage-600 mb-4">Preview</h3>
                <img
                  src={preview}
                  alt="Preview"
                  className="w-full rounded-lg border-2 border-heritage-200 shadow-md"
                />
              </div>
            )}
          </div>
        </div>

        {/* Action Button */}
        {preview && (
          <div className="mt-8 text-center">
            <button
              onClick={handleProceed}
              className="btn-primary inline-flex items-center space-x-3 text-lg"
            >
              <span>Proceed to Restoration</span>
              <FaArrowRight />
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default Upload;

