// Authentication utility functions
class AuthManager {
    static TOKEN_KEY = 'receipt_auth_token';
    static USER_KEY = 'receipt_user_data';

    static setToken(token) {
        localStorage.setItem(this.TOKEN_KEY, token);
    }

    static getToken() {
        return localStorage.getItem(this.TOKEN_KEY);
    }

    static setUserData(userData) {
        localStorage.setItem(this.USER_KEY, JSON.stringify(userData));
    }

    static getUserData() {
        const data = localStorage.getItem(this.USER_KEY);
        return data ? JSON.parse(data) : null;
    }

    static clearAuth() {
        localStorage.removeItem(this.TOKEN_KEY);
        localStorage.removeItem(this.USER_KEY);
    }

    static isAuthenticated() {
        return !!this.getToken();
    }

    static redirectToLogin() {
        window.location.href = '/login-page';
    }

    static redirectToHome() {
        window.location.href = '/';
    }

    static setupAxiosAuth() {
        const token = this.getToken();
        if (token) {
            axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
        }
    }

    static logout() {
        this.clearAuth();
        this.redirectToLogin();
    }

    static async checkAuthAndRedirect() {
        if (!this.isAuthenticated()) {
            this.redirectToLogin();
            return false;
        }
        this.setupAxiosAuth();
        return true;
    }
}

// Setup axios interceptor for handling 401 responses
axios.interceptors.response.use(
    response => response,
    error => {
        if (error.response && error.response.status === 401) {
            AuthManager.clearAuth();
            AuthManager.redirectToLogin();
        }
        return Promise.reject(error);
    }
);

// Setup axios to include auth header
AuthManager.setupAxiosAuth();
