import React, { createContext, useContext, useState } from 'react';

const ManuscriptContext = createContext();

export const useManuscript = () => {
  const context = useContext(ManuscriptContext);
  if (!context) {
    throw new Error('useManuscript must be used within ManuscriptProvider');
  }
  return context;
};

export const ManuscriptProvider = ({ children }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [restoredImage, setRestoredImage] = useState(null);
  const [extractedText, setExtractedText] = useState('');
  const [translations, setTranslations] = useState({
    english: '',
    hindi: '',
    kannada: ''
  });
  const [metadata, setMetadata] = useState({
    fileName: '',
    uploadDate: null,
    script: 'Devanagari',
    language: 'Sanskrit'
  });
  const [history, setHistory] = useState([]);

  const resetWorkflow = () => {
    setCurrentStep(0);
    setUploadedImage(null);
    setRestoredImage(null);
    setExtractedText('');
    setTranslations({ english: '', hindi: '', kannada: '' });
    setMetadata({
      fileName: '',
      uploadDate: null,
      script: 'Devanagari',
      language: 'Sanskrit'
    });
  };

  const saveToHistory = () => {
    const entry = {
      id: Date.now(),
      fileName: metadata.fileName,
      date: new Date().toISOString(),
      script: metadata.script,
      language: metadata.language,
      extractedText,
      translations,
      thumbnail: uploadedImage
    };
    setHistory(prev => [entry, ...prev]);
  };

  const value = {
    currentStep,
    setCurrentStep,
    uploadedImage,
    setUploadedImage,
    restoredImage,
    setRestoredImage,
    extractedText,
    setExtractedText,
    translations,
    setTranslations,
    metadata,
    setMetadata,
    history,
    setHistory,
    resetWorkflow,
    saveToHistory
  };

  return (
    <ManuscriptContext.Provider value={value}>
      {children}
    </ManuscriptContext.Provider>
  );
};

