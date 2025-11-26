import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import './index.css'
import AdminPage from './AdminPage'
import UserPage from './UserPage'

ReactDOM.createRoot(document.getElementById('root')).render(
  <BrowserRouter>
    <Routes>
      <Route path="/admin" element={<AdminPage />} />
      <Route path="/user" element={<UserPage />} />
      <Route path="/" element={<div className="p-10 text-xl">주소창에 /admin 또는 /user를 입력하세요.</div>} />
    </Routes>
  </BrowserRouter>,
)