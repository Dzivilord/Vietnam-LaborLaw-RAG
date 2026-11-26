"use client";

import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import { SendIcon } from "@/components/Icons";

const MAX_LENGTH = 4000; // matches the backend ChatRequest limit

interface ComposerProps {
  onSubmit: (query: string) => void;
  disabled: boolean;
  autoFocus?: boolean;
}

export default function Composer({ onSubmit, disabled, autoFocus }: ComposerProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Grow with the content up to ~6 lines.
  useEffect(() => {
    const element = textareaRef.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 168)}px`;
  }, [value]);

  useEffect(() => {
    if (autoFocus) textareaRef.current?.focus();
  }, [autoFocus]);

  function submit() {
    const query = value.trim();
    if (!query || disabled) return;
    onSubmit(query);
    setValue("");
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    submit();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submit();
    }
  }

  const nearLimit = value.length > MAX_LENGTH * 0.9;

  return (
    <div className="composer-wrap">
      <form className="composer" onSubmit={handleSubmit}>
        <label htmlFor="query" className="sr-only">Câu hỏi về luật lao động</label>
        <textarea
          id="query"
          ref={textareaRef}
          value={value}
          maxLength={MAX_LENGTH}
          rows={1}
          placeholder="Nhập câu hỏi về luật lao động…"
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button className="send" type="submit" disabled={disabled || !value.trim()} aria-label="Gửi câu hỏi">
          <SendIcon />
        </button>
      </form>
      <p className="composer-note">
        {nearLimit ? `${value.length}/${MAX_LENGTH} ký tự · ` : "Enter để gửi, Shift + Enter để xuống dòng · "}
        Thông tin chỉ mang tính tham khảo, không thay thế tư vấn pháp lý.
      </p>
    </div>
  );
}
