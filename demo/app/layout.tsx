import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ColdLens · 冷启动推荐实验室",
  description: "用匿名 Validation 样例解释短视频 Text-only 冷启动推荐。",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
