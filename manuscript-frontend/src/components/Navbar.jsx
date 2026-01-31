import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { FaScroll, FaHome, FaUpload, FaHistory, FaInfoCircle } from 'react-icons/fa';

const Navbar = () => {
  const location = useLocation();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const navLinks = [
    { path: '/', label: 'Home', icon: FaHome },
    { path: '/upload', label: 'Upload', icon: FaUpload },
    { path: '/history', label: 'Archive', icon: FaHistory },
    { path: '/about', label: 'About', icon: FaInfoCircle },
  ];

  const isActive = (path) => {
    if (path === '/') return location.pathname === '/';
    return location.pathname.startsWith(path);
  };

  return (
    <nav className={`sticky top-0 z-50 transition-all duration-300 ${
      scrolled 
        ? 'bg-white/80 backdrop-blur-lg shadow-xl border-b-2 border-saffron-400/30' 
        : 'bg-white/95 backdrop-blur-sm border-b-2 border-saffron-400/50 shadow-lg'
    }`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-20">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-3 group">
            <div className="relative">
              <FaScroll className="text-4xl text-saffron-500 group-hover:text-saffron-600 transition-all group-hover:rotate-12 transform" />
              <div className="absolute -inset-1 bg-saffron-400/20 rounded-full blur group-hover:bg-saffron-500/30 transition-all"></div>
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-heritage-600 to-saffron-600 bg-clip-text text-transparent group-hover:from-heritage-700 group-hover:to-saffron-700 transition-all">
                Sanskrit Manuscript
              </h1>
              <p className="text-xs text-heritage-400 -mt-1 font-medium">Digital Preservation</p>
            </div>
          </Link>

          {/* Navigation Links */}
          <div className="hidden md:flex space-x-2">
            {navLinks.map((link) => {
              const Icon = link.icon;
              return (
                <Link
                  key={link.path}
                  to={link.path}
                  className={`flex items-center space-x-2 px-5 py-2.5 rounded-xl font-semibold transition-all duration-300 relative group ${
                    isActive(link.path)
                      ? 'bg-gradient-to-r from-saffron-400 to-saffron-500 text-white shadow-lg'
                      : 'text-heritage-600 hover:bg-parchment-100'
                  }`}
                >
                  <Icon className={`text-lg ${isActive(link.path) ? 'animate-pulse' : 'group-hover:scale-110 transition-transform'}`} />
                  <span>{link.label}</span>
                  {!isActive(link.path) && (
                    <div className="absolute inset-0 bg-gradient-to-r from-saffron-400 to-saffron-500 rounded-xl opacity-0 group-hover:opacity-10 transition-opacity"></div>
                  )}
                </Link>
              );
            })}
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden">
            <button className="text-heritage-600 hover:text-heritage-700 p-2 rounded-lg hover:bg-parchment-100 transition-all">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;

