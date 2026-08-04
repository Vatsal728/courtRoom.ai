import React, { useState, useRef, useEffect, useCallback } from 'react';
import type { Language, LanguageInfo } from '../lib/types';
import { searchLanguages, getLanguageInfo, LANGUAGES } from '../lib/language';

interface LanguageDropdownProps {
  value: Language;
  onChange: (lang: Language) => void;
  placeholder?: string;
  className?: string;
  buttonClassName?: string;
}

export function LanguageDropdown({
  value,
  onChange,
  placeholder = 'Select language',
  className = '',
  buttonClassName = '',
}: LanguageDropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [highlightedIndex, setHighlightedIndex] = useState(0);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const filteredLanguages = searchLanguages(searchQuery);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
        setSearchQuery('');
        setHighlightedIndex(0);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    if (isOpen) {
      setSearchQuery('');
      setHighlightedIndex(0);
      inputRef.current?.focus();
    }
  }, [isOpen]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!isOpen) return;
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setHighlightedIndex(prev => Math.min(prev + 1, filteredLanguages.length - 1));
        break;
      case 'ArrowUp':
        e.preventDefault();
        setHighlightedIndex(prev => Math.max(prev - 1, 0));
        break;
      case 'Enter':
        e.preventDefault();
        if (filteredLanguages[highlightedIndex]) {
          onChange(filteredLanguages[highlightedIndex].code);
          setIsOpen(false);
          setSearchQuery('');
          setHighlightedIndex(0);
        }
        break;
      case 'Escape':
        setIsOpen(false);
        setSearchQuery('');
        setHighlightedIndex(0);
        break;
      default:
        break;
    }
  };

  const handleSearch = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
    setHighlightedIndex(0);
  };

  const handleSelect = (lang: LanguageInfo) => {
    onChange(lang.code);
    setIsOpen(false);
    setSearchQuery('');
    setHighlightedIndex(0);
  };

  const currentLangInfo = getLanguageInfo(value);

  return (
    <div ref={dropdownRef} className={`relative ${className}`} onKeyDown={handleKeyDown}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center gap-2 px-3 py-2 bg-slate-800/50 border border-slate-700 rounded-lg text-sm text-white hover:border-amber-400/50 focus:outline-none focus:ring-2 focus:ring-amber-400/50 transition-all min-w-[160px] ${buttonClassName}`}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
      >
        <span className="font-medium">{currentLangInfo.native}</span>
        <span className="text-xs text-slate-400 px-1.5 py-0.5 rounded bg-slate-700/50">
          {currentLangInfo.name}
        </span>
        <svg
          className={`ml-auto w-4 h-4 text-slate-400 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <div className="absolute z-50 mt-1.5 w-full max-h-64 overflow-auto bg-slate-800/95 backdrop-blur-xl border border-slate-700 rounded-lg shadow-xl">
          <input
            ref={inputRef}
            type="text"
            placeholder="Search languages..."
            value={searchQuery}
            onChange={handleSearch}
            className="w-full px-3 py-2 border-b border-slate-700 bg-slate-900/50 text-white placeholder-slate-500 text-sm focus:outline-none"
            aria-label="Search languages"
          />
          <ul role="listbox" className="py-1 max-h-56 overflow-auto">
            {filteredLanguages.length === 0 ? (
              <li className="px-3 py-2 text-slate-500 text-sm text-center">No languages found</li>
            ) : (
              filteredLanguages.map((lang, index) => (
                <li
                  key={lang.code}
                  role="option"
                  aria-selected={index === highlightedIndex}
                  onClick={() => handleSelect(lang)}
                  onMouseEnter={() => setHighlightedIndex(index)}
                  className={`px-3 py-2 text-sm cursor-pointer flex items-center gap-3 transition-colors ${
                    index === highlightedIndex
                      ? 'bg-amber-500/20 text-amber-300'
                      : 'text-white hover:bg-slate-700/50'
                  }`}
                >
                  <span className="font-medium min-w-[100px]">{lang.native}</span>
                  <span className="text-xs text-slate-400 px-1.5 py-0.5 rounded bg-slate-700/50">
                    {lang.name}
                  </span>
                  <span className="text-xs text-slate-500 ml-auto">{lang.code}</span>
                </li>
              ))
            )}
          </ul>
        </div>
      )}
    </div>
  );
}