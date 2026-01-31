import React from 'react';
import { useManuscript } from '../contexts/ManuscriptContext';
import { FaHistory, FaDownload, FaEye, FaTrash } from 'react-icons/fa';
import { toast } from 'react-toastify';

const History = () => {
  const { history, setHistory } = useManuscript();

  const handleDelete = (id) => {
    setHistory(history.filter(item => item.id !== id));
    toast.success('Entry deleted from archive');
  };

  const handleDownload = (entry) => {
    const content = `
Sanskrit Manuscript Translation
================================

File: ${entry.fileName}
Date: ${new Date(entry.date).toLocaleString()}
Script: ${entry.script}
Language: ${entry.language}

Original Sanskrit Text:
${entry.extractedText}

English Translation:
${entry.translations.english}

हिंदी अनुवाद:
${entry.translations.hindi}

ಕನ್ನಡ ಅನುವಾದ:
${entry.translations.kannada}
    `;

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${entry.fileName}-translation.txt`;
    link.click();
    toast.success('Translation downloaded!');
  };

  return (
    <div className="min-h-screen bg-parchment-100 py-8 page-transition overflow-hidden max-w-full">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 overflow-hidden">
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-heritage-600 mb-3 flex items-center justify-center">
            <FaHistory className="mr-4 text-saffron-500" />
            Manuscript Archive
          </h1>
          <p className="text-lg text-heritage-400">
            View and manage your previously processed manuscripts
          </p>
        </div>

        {history.length === 0 ? (
          <div className="card text-center py-20">
            <FaHistory className="text-6xl text-heritage-300 mx-auto mb-6" />
            <h2 className="text-2xl font-bold text-heritage-600 mb-3">No Manuscripts Yet</h2>
            <p className="text-heritage-400 mb-6">
              Your processed manuscripts will appear here for future reference
            </p>
            <a href="/upload" className="btn-primary inline-block">
              Upload Your First Manuscript
            </a>
          </div>
        ) : (
          <div className="space-y-6">
            {history.map((entry) => (
              <div key={entry.id} className="card hover:shadow-heritage-lg transition-all duration-300">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                  {/* Thumbnail */}
                  <div className="md:col-span-1">
                    <img
                      src={entry.thumbnail}
                      alt={entry.fileName}
                      className="w-full h-40 object-cover rounded-lg border-2 border-heritage-200"
                    />
                  </div>

                  {/* Details */}
                  <div className="md:col-span-2">
                    <h3 className="text-xl font-bold text-heritage-600 mb-2">{entry.fileName}</h3>
                    <div className="space-y-1 text-sm text-heritage-500">
                      <p>
                        <span className="font-semibold">Date:</span> {new Date(entry.date).toLocaleString()}
                      </p>
                      <p>
                        <span className="font-semibold">Script:</span> {entry.script}
                      </p>
                      <p>
                        <span className="font-semibold">Language:</span> {entry.language}
                      </p>
                    </div>
                    <div className="mt-4">
                      <p className="text-sm text-heritage-400 line-clamp-3">
                        {entry.extractedText}
                      </p>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="md:col-span-1 flex flex-col justify-center space-y-3">
                    <button
                      onClick={() => handleDownload(entry)}
                      className="btn-secondary w-full inline-flex items-center justify-center space-x-2 text-sm"
                    >
                      <FaDownload />
                      <span>Download</span>
                    </button>
                    <button
                      onClick={() => handleDelete(entry.id)}
                      className="bg-red-100 hover:bg-red-200 text-red-700 font-semibold py-2 px-4 rounded-lg transition-all duration-300 w-full inline-flex items-center justify-center space-x-2 text-sm"
                    >
                      <FaTrash />
                      <span>Delete</span>
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default History;

