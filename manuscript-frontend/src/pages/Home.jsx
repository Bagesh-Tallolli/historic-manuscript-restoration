import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { FaUpload, FaMagic, FaLanguage, FaBookOpen, FaArrowRight, FaStar, FaShieldAlt, FaGlobe } from 'react-icons/fa';

const Home = () => {
  const features = [
    {
      icon: FaUpload,
      title: 'Upload Manuscript',
      description: 'Support for PNG, JPG, and JPEG formats with drag-and-drop interface.',
      color: 'from-blue-400 to-blue-600',
    },
    {
      icon: FaMagic,
      title: 'Image Restoration',
      description: 'Advanced CLAHE and unsharp mask techniques for enhanced clarity.',
      color: 'from-purple-400 to-purple-600',
    },
    {
      icon: FaBookOpen,
      title: 'OCR Extraction',
      description: 'AI-powered Sanskrit text recognition from manuscript images.',
      color: 'from-green-400 to-green-600',
    },
    {
      icon: FaLanguage,
      title: 'Multi-language Translation',
      description: 'Accurate translations to English, Hindi, and Kannada.',
      color: 'from-orange-400 to-orange-600',
    },
  ];

  const workflow = [
    { step: '1', title: 'Upload', desc: 'Select your manuscript image', icon: FaUpload },
    { step: '2', title: 'Restore', desc: 'Enhance image quality automatically', icon: FaMagic },
    { step: '3', title: 'Extract', desc: 'OCR extracts Sanskrit text', icon: FaBookOpen },
    { step: '4', title: 'Translate', desc: 'Get multi-language translations', icon: FaLanguage },
  ];

  const stats = [
    { icon: FaStar, value: '99%', label: 'Accuracy' },
    { icon: FaShieldAlt, value: '100%', label: 'Secure' },
    { icon: FaGlobe, value: '3+', label: 'Languages' },
  ];

  return (
    <div className="min-h-screen page-transition overflow-hidden max-w-full">
      {/* Hero Section with Enhanced Design */}
      <section className="relative bg-gradient-to-br from-amber-50 via-orange-50 to-yellow-50 py-24 overflow-hidden">
        {/* Decorative elements */}
        <div className="absolute top-0 left-0 w-full h-full opacity-10 overflow-hidden">
          <div className="absolute top-20 left-10 w-64 h-64 bg-saffron-400 rounded-full blur-3xl"></div>
          <div className="absolute bottom-20 right-10 w-96 h-96 bg-heritage-400 rounded-full blur-3xl"></div>
        </div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10 overflow-hidden">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            {/* Floating badge */}
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2 }}
              className="inline-block mb-6"
            >
              <span className="px-4 py-2 bg-white/80 backdrop-blur-sm rounded-full text-sm font-semibold text-heritage-600 shadow-lg border border-saffron-200">
                ✨ AI-Powered Heritage Preservation
              </span>
            </motion.div>

            <h1 className="text-6xl md:text-7xl font-bold mb-6 leading-tight">
              <span className="gradient-text">Sanskrit Manuscript</span>
              <br />
              <span className="text-heritage-600">Restoration & Translation</span>
            </h1>

            <div className="decorative-line"></div>

            <p className="text-2xl text-heritage-500 mb-8 max-w-3xl mx-auto font-serif italic">
              Preserving Ancient Wisdom Through Modern Technology
            </p>
            <p className="text-lg text-heritage-400 mb-12 max-w-2xl mx-auto leading-relaxed">
              A digital heritage initiative combining AI-powered OCR and translation to make ancient Sanskrit manuscripts accessible to scholars and researchers worldwide.
            </p>

            <div className="flex flex-col sm:flex-row gap-6 justify-center items-center">
              <Link to="/upload" className="btn-primary inline-flex items-center justify-center space-x-3 text-lg group">
                <FaUpload className="group-hover:rotate-12 transition-transform" />
                <span>Start Processing</span>
                <FaArrowRight className="group-hover:translate-x-1 transition-transform" />
              </Link>
              <Link to="/about" className="btn-secondary inline-flex items-center justify-center space-x-2 text-lg">
                <span>Learn More</span>
              </Link>
            </div>

            {/* Stats */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
              className="grid grid-cols-3 gap-8 max-w-2xl mx-auto mt-16"
            >
              {stats.map((stat, index) => (
                <div key={index} className="text-center">
                  <stat.icon className="text-3xl text-saffron-500 mx-auto mb-2" />
                  <div className="text-3xl font-bold text-heritage-700">{stat.value}</div>
                  <div className="text-sm text-heritage-500">{stat.label}</div>
                </div>
              ))}
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* How It Works - Enhanced */}
      <section className="py-20 bg-white relative overflow-hidden">
        <div className="section-divider"></div>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 overflow-hidden">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-5xl font-bold text-heritage-600 mb-4">
              How It Works
            </h2>
            <p className="text-xl text-heritage-400">Simple 4-step process to digitize your manuscripts</p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 relative">
            {/* Connection lines */}
            <div className="hidden md:block absolute top-1/4 left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-saffron-200 to-transparent"></div>

            {workflow.map((item, index) => {
              const IconComponent = item.icon;
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.15 }}
                  viewport={{ once: true }}
                  className="relative"
                >
                  <div className="card-elevated text-center hover-lift group">
                    <div className="relative mb-6">
                      <div className="w-20 h-20 bg-gradient-to-br from-saffron-400 to-saffron-600 rounded-2xl flex items-center justify-center text-3xl font-bold text-white mx-auto shadow-2xl group-hover:rotate-6 transition-transform">
                        {item.step}
                      </div>
                      <div className="absolute -top-2 -right-2">
                        <IconComponent className="text-2xl text-heritage-600 bg-white rounded-full p-2 shadow-lg" />
                      </div>
                    </div>
                    <h3 className="text-2xl font-bold text-heritage-600 mb-3">{item.title}</h3>
                    <p className="text-heritage-400 leading-relaxed">{item.desc}</p>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Features - Enhanced */}
      <section className="py-20 bg-gradient-to-br from-parchment-100 via-parchment-50 to-white overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 overflow-hidden">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-5xl font-bold text-heritage-600 mb-4">
              Key Features
            </h2>
            <div className="decorative-line"></div>
            <p className="text-xl text-heritage-400">Advanced tools for scholarly manuscript digitization</p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => {
              const IconComponent = feature.icon;
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="group"
                >
                  <div className="card hover-lift hover-glow text-center h-full">
                    <div className={`w-16 h-16 bg-gradient-to-br ${feature.color} rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-xl group-hover:scale-110 transition-transform`}>
                      <IconComponent className="text-3xl text-white" />
                    </div>
                    <h3 className="text-xl font-bold text-heritage-600 mb-3">{feature.title}</h3>
                    <p className="text-heritage-400 text-sm leading-relaxed">{feature.description}</p>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* CTA Section - Enhanced */}
      <section className="relative py-24 bg-gradient-to-br from-heritage-600 via-heritage-700 to-heritage-800 overflow-hidden">
        {/* Decorative background */}
        <div className="absolute inset-0 opacity-10 overflow-hidden">
          <div className="absolute top-0 left-0 w-full h-full" style={{
            backgroundImage: 'radial-gradient(circle at 2px 2px, white 1px, transparent 0)',
            backgroundSize: '40px 40px'
          }}></div>
        </div>

        <div className="max-w-5xl mx-auto text-center px-4 relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-5xl md:text-6xl font-bold text-white mb-6">
              Ready to Preserve Heritage?
            </h2>
            <div className="w-24 h-1 bg-saffron-400 mx-auto mb-8"></div>
            <p className="text-2xl text-parchment-100 mb-12 max-w-3xl mx-auto leading-relaxed">
              Start digitizing your Sanskrit manuscripts today with our AI-powered platform
            </p>
            <div className="flex flex-col sm:flex-row gap-6 justify-center">
              <Link to="/upload" className="group inline-flex items-center space-x-3 bg-gradient-to-r from-saffron-400 to-saffron-500 hover:from-saffron-500 hover:to-saffron-600 text-white font-bold py-5 px-10 rounded-2xl transition-all duration-300 shadow-2xl hover:shadow-saffron-500/50 text-lg">
                <FaUpload className="group-hover:rotate-12 transition-transform" />
                <span>Upload Your First Manuscript</span>
                <FaArrowRight className="group-hover:translate-x-1 transition-transform" />
              </Link>
              <Link to="/history" className="inline-flex items-center space-x-3 bg-white/10 backdrop-blur-sm hover:bg-white/20 text-white font-bold py-5 px-10 rounded-2xl transition-all duration-300 border-2 border-white/30 text-lg">
                <span>View Archive</span>
              </Link>
            </div>

            {/* Trust indicators */}
            <div className="mt-16 grid grid-cols-3 gap-8 max-w-3xl mx-auto">
              <div className="text-center">
                <div className="text-4xl font-bold text-saffron-400 mb-2">Fast</div>
                <div className="text-sm text-parchment-200">Processing in seconds</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-saffron-400 mb-2">Accurate</div>
                <div className="text-sm text-parchment-200">99% OCR precision</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-saffron-400 mb-2">Secure</div>
                <div className="text-sm text-parchment-200">Your data protected</div>
              </div>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default Home;

