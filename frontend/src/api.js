import axios from "axios";

const BASE = "http://localhost:8000";

export const uploadResume = (file) => {
  const formData = new FormData();
  formData.append("file", file);
  return axios.post(`${BASE}/upload/`, formData);
};

export const runAnalysis = () => {
  return axios.post(`${BASE}/run/`);
};
