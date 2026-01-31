import React from 'react';
import { useManuscript } from '../contexts/ManuscriptContext';
import { FaCheck } from 'react-icons/fa';

const StepProgressBar = () => {
  const { currentStep } = useManuscript();

  const steps = [
    { number: 1, label: 'Upload', path: '/upload' },
    { number: 2, label: 'Restore', path: '/restore' },
    { number: 3, label: 'Extract OCR', path: '/ocr' },
    { number: 4, label: 'Translate', path: '/translate' },
  ];

  return (
    <div className="w-full max-w-4xl mx-auto py-8">
      <div className="flex items-center justify-between">
        {steps.map((step, index) => (
          <React.Fragment key={step.number}>
            {/* Step Circle */}
            <div className="flex flex-col items-center">
              <div
                className={`w-12 h-12 rounded-full flex items-center justify-center font-bold text-lg transition-all duration-300 ${
                  currentStep > index
                    ? 'bg-saffron-500 text-white shadow-lg'
                    : currentStep === index
                    ? 'bg-saffron-400 text-heritage-700 shadow-lg ring-4 ring-saffron-200'
                    : 'bg-parchment-300 text-heritage-400'
                }`}
              >
                {currentStep > index ? <FaCheck /> : step.number}
              </div>
              <span
                className={`mt-2 text-sm font-medium ${
                  currentStep >= index ? 'text-heritage-700' : 'text-heritage-400'
                }`}
              >
                {step.label}
              </span>
            </div>

            {/* Connector Line */}
            {index < steps.length - 1 && (
              <div className="flex-1 h-1 mx-2 relative top-[-20px]">
                <div
                  className={`h-full rounded transition-all duration-500 ${
                    currentStep > index ? 'bg-saffron-500' : 'bg-parchment-300'
                  }`}
                />
              </div>
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};

export default StepProgressBar;

