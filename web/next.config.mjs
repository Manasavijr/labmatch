const API = process.env.LABMATCH_API || 'http://127.0.0.1:8000';
export default {
  async rewrites() {
    return [{ source: '/api/:path*', destination: `${API}/api/:path*` }];
  },
};
