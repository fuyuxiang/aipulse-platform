import { Button, Result } from "antd";
import React from "react";

export class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError(): { failed: boolean } {
    return { failed: true };
  }
  render(): React.ReactNode {
    return this.state.failed ? (
      <Result
        status="error"
        title="页面加载失败"
        subTitle="请重新加载页面。如果问题持续，请联系管理员。"
        extra={
          <Button type="primary" onClick={() => window.location.reload()}>
            重新加载
          </Button>
        }
      />
    ) : (
      this.props.children
    );
  }
}
