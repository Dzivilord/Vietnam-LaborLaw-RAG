import "./globals.css";

export const metadata = {
  title: "Trợ lý pháp luật lao động",
  description: "Tra cứu và kiểm chứng thông tin từ kho dữ liệu pháp luật lao động Việt Nam.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="vi">
      <body>{children}</body>
    </html>
  );
}
