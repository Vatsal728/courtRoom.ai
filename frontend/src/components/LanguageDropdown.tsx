import React, { useState, useRef, useEffect } from 'react';
import type { Language, LanguageInfo } from '../lib/types';
import { searchLanguages, getLanguageInfo } from '../lib/language';

interface LanguageDropdownProps {
  value: Language;
  onChange: (lang: Language) => void;
  className?: string;
  buttonClassName?: string;
}

export function LanguageDropdown({
  value,
  onChange,
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
  const isEnglish = value === 'en';

  return (
    <div ref={dropdownRef} className={`relative ${className}`} onKeyDown={handleKeyDown}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center gap-1.5 px-2.5 py-1.5 bg-card border border-line rounded-lg text-xs text-ink hover:bg-hover hover:border-[#cfcfcf] dark:hover:border-[#3a4454] focus:outline-none focus:ring-2 focus:ring-black/[0.04] dark:focus:ring-white/[0.04] transition-all min-w-[96px] ${buttonClassName}`}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        title={currentLangInfo.name}
      >
        <span className="font-semibold leading-tight truncate">
          {isEnglish ? currentLangInfo.name : `${currentLangInfo.native} · ${currentLangInfo.name}`}
        </span>
        <svg
          className={`ml-auto w-3.5 h-3.5 text-ink-3 flex-shrink-0 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <div className="absolute z-50 mt-1.5 w-44 max-h-64 overflow-auto bg-card border border-line rounded-lg shadow-pop dropdown-fade-in">
          <input
            ref={inputRef}
            type="text"
            placeholder="Search languages..."
            value={searchQuery}
            onChange={handleSearch}
            className="w-full px-3 py-2 border-b border-line-2 bg-bg text-ink placeholder-ink-3 text-xs focus:outline-none rounded-t-lg"
            aria-label="Search languages"
          />
          <ul role="listbox" className="py-1 max-h-56 overflow-auto">
            {filteredLanguages.length === 0 ? (
              <li className="px-3 py-2 text-ink-3 text-xs text-center">No languages found</li>
            ) : (
              filteredLanguages.map((lang, index) => (
                <li
                  key={lang.code}
                  role="option"
                  aria-selected={index === highlightedIndex}
                  onClick={() => handleSelect(lang)}
                  onMouseEnter={() => setHighlightedIndex(index)}
                  className={`px-3 py-1.5 text-xs cursor-pointer flex items-center gap-2 transition-colors ${
                    index === highlightedIndex
                      ? 'bg-active text-ink'
                      : 'text-ink-2 hover:bg-hover'
                  }`}
                >
                  <span className="font-semibold min-w-[52px]">{lang.native}</span>
                  <span className="text-ink-3 truncate">{lang.name}</span>
                </li>
              ))
            )}
          </ul>
        </div>
      )}
    </div>
  );
}
