/**
 * auth.tsx - Authentication context and hooks
 * Handle user login, registration, and session management
 */

import React, { createContext, useState } from 'react';
import { loginUser, logoutUser, registerUser } from './api';

export interface AuthContextType {
  userId: string | null;
  userName: string | null;
  email: string | null;
  isLoggedIn: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, name: string) => Promise<void>;
  logout: () => void;
}

export const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [userId, setUserId] = useState<string | null>(() => localStorage.getItem('userId'));
  const [userName, setUserName] = useState<string | null>(() => localStorage.getItem('userName'));
  const [email, setEmail] = useState<string | null>(() => localStorage.getItem('userEmail'));

  const login = async (email: string, password: string) => {
    const result = await loginUser(email, password);
    setUserId(result.user_id);
    setUserName(result.name);
    setEmail(result.email);
  };

  const register = async (email: string, password: string, name: string) => {
    await registerUser(email, password, name);
    // Auto-login after register
    await login(email, password);
  };

  const logout = () => {
    logoutUser();
    setUserId(null);
    setUserName(null);
    setEmail(null);
  };

  return (
    <AuthContext.Provider value={{ userId, userName, email, isLoggedIn: !!userId, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = React.useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used within AuthProvider');
  return context;
}
