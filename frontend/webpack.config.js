const HtmlWebpackPlugin = require('html-webpack-plugin');
const path = require('path');

module.exports = {
  entry: './src/main.tsx',
  output: {
    path: path.resolve(__dirname, 'dist'),
    clean: true,
    filename: '[name].[contenthash].js',
    publicPath: '/'
  },
  optimization: { splitChunks: { chunks: 'all' }, runtimeChunk: 'single' },
  resolve: {
    extensions: ['.tsx', '.ts', '.js']
  },
  module: {
    rules: [
      { test: /\.tsx?$/, use: { loader: 'ts-loader', options: { transpileOnly: true } }, exclude: /node_modules/ },
      { test: /\.css$/, use: ['style-loader', 'css-loader', 'postcss-loader'] }
    ]
  },
  plugins: [
    new HtmlWebpackPlugin({
      templateContent: '<!doctype html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/><title>AIPulse</title></head><body><div id="root"></div></body></html>'
    })
  ],
  devServer: {
    historyApiFallback: true,
    proxy: [
      {
        context: ['/api', '/health'],
        target: process.env.AIPULSE_E2E_API_TARGET || 'http://127.0.0.1:8000',
        changeOrigin: true,
        ws: true
      }
    ]
  }
};
