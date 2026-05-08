import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const predictArticle = async (text) => {
    try {
        const response = await axios.post(`${API_BASE_URL}/predict`, { text });
        return response.data;
    } catch (error) {
        console.error("Error calling prediction API:", error);
        throw error;
    }
};
