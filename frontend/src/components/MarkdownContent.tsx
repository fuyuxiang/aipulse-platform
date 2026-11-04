import React from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

export function MarkdownContent({
  children,
}: {
  children: string;
}): JSX.Element {
  return (
    <div className="markdown-content">
      <Markdown
        skipHtml
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noopener noreferrer">
              {children}
            </a>
          ),
          img: ({ src, alt }) =>
            src ? (
              <a href={src} target="_blank" rel="noopener noreferrer">
                查看图片{alt ? `：${alt}` : ""}
              </a>
            ) : (
              <span>{alt}</span>
            ),
        }}
      >
        {children}
      </Markdown>
    </div>
  );
}
