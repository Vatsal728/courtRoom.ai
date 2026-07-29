import React, { useState, useRef, useCallback } from 'react';
import { submitQuery } from './lib/api';
import type { RetrievalResult } from './lib/types';

export default function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<RetrievalResult[]>([]);
  const [detectedLang, setDetectedLang] = useState('en');
  const [detectedCat, setDetectedCat] = useState('');
  const resultsRef = useRef<HTMLDivElement>(null);

  const handleSearch = useCallback(async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    
    try {
      // Get user ID from localStorage (set after login)
      const userId = localStorage.getItem('userId') || 'anonymous';
      
      // Call FastAPI backend
      const result = await submitQuery(query, userId);
      
      // Transform to RetrievalResult format
      const transformedResults: RetrievalResult[] = (result.sources || []).map((source: any, i: number) => ({
        section: {
          id: `${i}`,
          document_id: source.source_act,
          section_number: source.section,
          title: source.topic,
          content: source.content_preview,
          plain_language: source.content_preview,
          keywords: source.keywords,
          category: result.domain,
          next_steps: '',
          jurisdiction: source.courts[0],
          document_name: source.source_act,
          document_short_name: source.source_act.split(' ')[0],
        },
        score: 100 - (i * 10),
        matchedTerms: source.keywords,
      }));
      
      setResults(transformedResults);
      setDetectedLang('en');
      setDetectedCat(result.domain);
      
      // Store query in state for draft generation
      localStorage.setItem('lastQuery', query);
      localStorage.setItem('lastQueryId', result.query_id || '');
      
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Something went wrong');
    } finally {
      setLoading(false);
    }
  }, [query]);

  return (
    <div>
      {/* Component Layout */}
    </div>
  );
}
