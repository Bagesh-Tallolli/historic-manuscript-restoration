import React from 'react';
import { FaScroll } from 'react-icons/fa';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-gradient-to-br from-heritage-800 to-heritage-900 text-parchment-100 mt-20 relative overflow-hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 relative z-10">
        <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
          {/* Logo and Title */}
          <div className="flex items-center space-x-3">
            <FaScroll className="text-2xl text-saffron-400" />
            <div>
              <h3 className="text-lg font-bold text-white">Sanskrit Manuscript Restoration</h3>
              <p className="text-xs text-parchment-300">Digital Heritage Preservation</p>
            </div>
          </div>

          {/* Copyright */}
          <div className="text-center">
            <p className="text-sm text-parchment-300">
              © {currentYear} Sanskrit Manuscript Restoration
            </p>
            <p className="text-xs text-parchment-400 mt-1">
              Academic & Cultural Heritage Preservation
            </p>
          </div>
        </div>
      </div>

      {/* Bottom accent line */}
      <div className="h-1 bg-gradient-to-r from-saffron-400 via-heritage-500 to-saffron-400"></div>
    </footer>
  );
};

export default Footer;

