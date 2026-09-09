import "./globals.css";

export const metadata = {
  title: "Labor Law RAG",
  description: "Search and inspect the Vietnamese labor-law archive.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
