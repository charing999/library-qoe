import React, { useEffect, useState } from "react";
import axios from "axios";

export default function UserPage() {
  const [data, setData] = useState(null);

  useEffect(() => {
    axios.get("http://localhost:8000/api/recommend")
      .then((res) => setData(res.data));
  }, []);

  if (!data) return <div>Loading...</div>;

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center p-4">
      <div className="w-full max-w-md bg-white rounded-xl shadow-lg overflow-hidden">
        <div className="bg-blue-600 p-6 text-white text-center">
          <h1 className="text-xl font-bold">도서관 Wi-Fi 가이드</h1>
          <p className="text-blue-100 text-sm mt-1">AI가 추천하는 최적의 공부 장소</p>
        </div>

        <div className="p-6">
          <div className="mb-6 text-center">
            <p className="text-gray-500 text-sm mb-1">지금 가장 쾌적한 곳</p>
            <h2 className="text-2xl font-bold text-blue-600">{data.best_zone}</h2>
            <p className="text-sm text-blue-400 mt-1">"{data.message}"</p>
          </div>

          <div className="space-y-3">
            {data.zones.map((zone) => (
              <div key={zone.name} className="flex justify-between items-center p-3 bg-gray-50 rounded-lg border">
                <span className="font-medium">{zone.name}</span>
                <span className={`px-3 py-1 rounded-full text-xs font-bold text-white
                  ${zone.grade === "Good" ? "bg-green-500" : zone.grade === "Moderate" ? "bg-yellow-400" : "bg-red-500"}`}>
                  {zone.grade}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}