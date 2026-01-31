# 🎨 UI/UX Visual Guide - Sanskrit Manuscript Restoration

## 📱 Application Preview

### **Page Layout Overview**

```
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║        Sanskrit Manuscript Restoration & Translation          ║
║         Digital Preservation of Ancient Indian Manuscripts    ║
║                    ─────────────────────                      ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  📜 Upload Manuscript Image                                   ║
║  ══════════════════════════════════════════                   ║
║                                                                ║
║  [Browse Files...]                                            ║
║                                                                ║
║  ┌──────────────────────────────────────────────┐            ║
║  │                                               │            ║
║  │         Original Manuscript                   │            ║
║  │         [Image Display Area]                  │            ║
║  │                                               │            ║
║  └──────────────────────────────────────────────┘            ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  🧹 Image Restoration                                         ║
║  ══════════════════════════════════════════                   ║
║                                                                ║
║  [🧹 Restore Manuscript Image]  ← Button (Saffron colored)   ║
║                                                                ║
║  ┌────────────────────┐    ┌────────────────────┐           ║
║  │ Original           │    │ Restored           │           ║
║  │ Manuscript         │    │ Manuscript         │           ║
║  │ [Image]            │    │ [Image]            │           ║
║  └────────────────────┘    └────────────────────┘           ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  🔍 OCR Extraction                                            ║
║  ══════════════════════════════════════════                   ║
║                                                                ║
║  [🔍 Extract Sanskrit Text (OCR)]  ← Button                  ║
║                                                                ║
║  📖 Extracted Sanskrit Text                                   ║
║  ┌──────────────────────────────────────────────┐            ║
║  │                                               │            ║
║  │  Sanskrit text displayed here in             │            ║
║  │  Devanagari script with proper font          │            ║
║  │  and spacing                                  │            ║
║  │                                               │            ║
║  └──────────────────────────────────────────────┘            ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  🌐 Translation                                               ║
║  ══════════════════════════════════════════                   ║
║                                                                ║
║  [🌐 Translate Extracted Text]  ← Button                     ║
║                                                                ║
║  ┌──────────────────────────────────────────────┐            ║
║  │ **Extracted Sanskrit Text:**                  │            ║
║  │ [Sanskrit in Devanagari]                      │            ║
║  │                                               │            ║
║  │ **English Meaning:**                          │            ║
║  │ [English translation]                         │            ║
║  │                                               │            ║
║  │ **हिंदी अर्थ:**                                │            ║
║  │ [Hindi translation in Devanagari]             │            ║
║  │                                               │            ║
║  │ **ಕನ್ನಡ ಅರ್ಥ:**                                │            ║
║  │ [Kannada translation]                         │            ║
║  └──────────────────────────────────────────────┘            ║
║                                                                ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  Designed for Academic & Cultural Heritage Preservation       ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

---

## 🎨 Color Palette

```
Background:         #FFF8EE  (Ivory/Off-white)
                    ███████████

Primary Accent:     #F4C430  (Light Saffron)
                    ███████████

Secondary Accent:   #5A3E1B  (Deep Brown)
                    ███████████

Text Primary:       #2C1810  (Dark Brown)
                    ███████████

Card Background:    #FAF0E6  (Linen)
                    ███████████

Border Color:       #DEB887  (Burlywood)
                    ███████████
```

---

## 🔤 Typography

### **Main Title**
- **Font:** Crimson Text (Serif)
- **Size:** 3rem
- **Weight:** Bold (700)
- **Color:** #D2691E (Chocolate)
- **Alignment:** Center

### **Section Headers**
- **Font:** Crimson Text (Serif)
- **Size:** 1.8rem
- **Weight:** Semi-bold (600)
- **Color:** #8B4513 (Saddle Brown)
- **Border-bottom:** 2px solid #F4C430

### **Sanskrit Text**
- **Font:** Noto Serif Devanagari
- **Size:** 1.3rem
- **Line-height:** 2
- **Background:** #FFFAF0 (Parchment)

### **Translations**
- **Font:** System default with Unicode support
- **Size:** 1.1rem
- **Line-height:** 1.8
- **Alignment:** Justified

---

## 📐 Layout Specifications

### **Container Widths**
- Main content: `centered` (max 1000px)
- Image displays: `use_container_width=True`
- Cards: 100% with padding

### **Spacing**
- Section margins: 2rem top
- Card padding: 1.5rem
- Button padding: 0.7rem × 2rem
- Image borders: 2px solid

### **Border Radius**
- Cards: 12px
- Buttons: 8px
- Images: 8px
- Text areas: 8px

---

## 🎭 Interactive Elements

### **Buttons**
```
Normal State:
  Background: #F4C430 (Saffron)
  Text: #5A3E1B (Deep Brown)
  Border: 2px solid #D2691E

Hover State:
  Background: #D2691E (Chocolate)
  Text: #FFFFFF (White)
  Border: 2px solid #8B4513
```

### **Button Visibility Rules**
1. **Restore Button:** Appears after image upload
2. **OCR Button:** Appears after restoration
3. **Translate Button:** Appears after OCR extraction

### **Loading States**
- Spinner appears during processing
- Message: "Restoring manuscript..." / "Extracting Sanskrit text..." etc.

---

## 📱 Responsive Behavior

### **Desktop (>1000px)**
- Two-column layout for image comparison
- Full-width cards
- Comfortable reading width

### **Tablet/Mobile (<1000px)**
- Single-column layout
- Stacked images
- Touch-friendly buttons

---

## 🔍 User Interaction Flow

### **Step 1: Upload**
```
User Action:     Click "Browse Files"
System Response: File picker opens
User Action:     Select image file
System Response: Image displays with label
                 "Restore" button appears
```

### **Step 2: Restore**
```
User Action:     Click "🧹 Restore Manuscript Image"
System Response: Spinner shows "Restoring manuscript..."
                 Side-by-side comparison appears
                 "OCR Extract" button appears
```

### **Step 3: OCR**
```
User Action:     Click "🔍 Extract Sanskrit Text (OCR)"
System Response: Spinner shows "Extracting..."
                 Sanskrit text displayed in styled box
                 "Translate" button appears
```

### **Step 4: Translate**
```
User Action:     Click "🌐 Translate Extracted Text"
System Response: Full translation card displays
                 Structured output (Sanskrit → English → Hindi → Kannada)
```

---

## 🎨 Visual Hierarchy

### **Priority Levels**

**Level 1 (Most Prominent):**
- Main title
- Action buttons (Saffron colored)
- Processed images

**Level 2 (Secondary):**
- Section headers
- Translated text
- Image labels

**Level 3 (Tertiary):**
- Subtitle
- Helper text
- Footer

---

## 🛡️ Error Handling

### **Visual Feedback**

**Success:**
- ✓ Green checkmark with message
- Example: "✓ Restoration complete!"

**Error:**
- ❌ Red error message
- Example: "❌ Error: Failed to process image"

**Warning:**
- ⚠️ Yellow warning message
- Example: "⚠️ Please upload an image first"

---

## 🎯 Design Principles Applied

### **1. Scholarly Aesthetic**
- Serif fonts throughout
- Muted, warm color palette
- Generous white space
- Professional borders and shadows

### **2. Cultural Respect**
- Saffron color (संस्कृत heritage)
- Support for Devanagari and Kannada scripts
- Academic tone in all text
- Traditional color scheme

### **3. Minimal & Clean**
- No technical jargon visible
- Clear visual hierarchy
- One action at a time
- No clutter or debug info

### **4. Academic Standards**
- University-grade appearance
- Suitable for research use
- Professional footer
- Structured output format

---

## 📊 Component Breakdown

### **Header Section**
```html
<div class="main-title">
  Sanskrit Manuscript Restoration & Translation
</div>
<div class="subtitle">
  Digital Preservation of Ancient Indian Manuscripts
</div>
```

### **Image Container**
```html
<div class="image-label">Original Manuscript</div>
<div class="image-container">
  [Image Display]
</div>
```

### **Translation Card**
```html
<div class="card">
  <div class="translation-title">**Extracted Sanskrit Text:**</div>
  <div class="translation-text">[Content]</div>
  
  <div class="translation-title">**English Meaning:**</div>
  <div class="translation-text">[Content]</div>
  
  <!-- Hindi and Kannada sections follow -->
</div>
```

---

## ✅ Accessibility Features

- High contrast text (WCAG AA compliant)
- Clear visual hierarchy
- Readable font sizes (minimum 1.1rem)
- Descriptive button labels
- Proper heading structure

---

## 🎓 Academic Appropriateness

### **✅ Suitable For:**
- University research projects
- Digital heritage initiatives
- Museum digitization programs
- Academic publications
- Cultural preservation institutions

### **❌ Removed (Non-Academic):**
- Gaming-style UI elements
- Flashy animations
- Technical debugging output
- Developer-oriented controls
- Casual language

---

**This design creates a dignified, scholarly environment appropriate for working with ancient Sanskrit manuscripts while maintaining modern usability standards.**

