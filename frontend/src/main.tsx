import React from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import "./styles.css";
import { App } from "./app";

const root = createRoot(document.getElementById("root") as HTMLElement);
const router = createBrowserRouter([{ path: "*", element: <App /> }]);
root.render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
);
