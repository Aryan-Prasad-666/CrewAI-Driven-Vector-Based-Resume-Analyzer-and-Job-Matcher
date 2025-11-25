import React, { useState } from "react";
import { uploadResume, runAnalysis } from "./api";
import {
  Container, Card, Button, Form, Alert, Spinner, Tabs, Tab, Badge
} from "react-bootstrap";
import ReactMarkdown from "react-markdown";
import ReactJson from "react-json-view";

export default function ResumeForm() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleRun = async () => {
    if (!file) return alert("Please upload your resume first.");
    setLoading(true);
    setError("");
    try {
      await uploadResume(file);
      const res = await runAnalysis();
      if (res.data.error) {
        setError(res.data.error);
        setResults({});
      } else {
        setResults(res.data);
      }
    } catch (err) {
      console.error(err);
      setError("❌ Something went wrong running the analysis.");
    } finally {
      setLoading(false);
    }
  };

  const download = (filename, content) => {
    const blob = new Blob([content], { type: "text/plain" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
    URL.revokeObjectURL(link.href);
  };

  return (
    <Container className="py-4">
      <Card className="p-4 shadow-sm">
        <h3 className="text-center mb-4 text-primary">Resume AI Assistant</h3>

        <Form.Group controlId="upload" className="mb-3">
          <Form.Label>Upload Resume (PDF)</Form.Label>
          <Form.Control
            type="file"
            accept=".pdf"
            onChange={(e) => setFile(e.target.files[0])}
          />
        </Form.Group>

        <div className="text-center mb-4">
          <Button onClick={handleRun} disabled={loading}>
            {loading ? (
              <>
                <Spinner animation="border" size="sm" /> Processing...
              </>
            ) : (
              "Run Analysis"
            )}
          </Button>
        </div>

        {error && <Alert variant="danger">{error}</Alert>}

        {Object.keys(results).length > 0 && (
          <Tabs defaultActiveKey="resume_summary" className="mt-3">
            {Object.entries(results).map(([key, value]) => (
              <Tab eventKey={key} title={formatTabName(key)} key={key}>
                <div className="mt-3">
                  {renderTabContent(key, value)}
                  <div className="mt-3 text-end">
                    <Button
                      size="sm"
                      variant="outline-secondary"
                      onClick={() => download(`${key}.txt`, value)}
                    >
                      ⬇️ Download
                    </Button>
                  </div>
                </div>
              </Tab>
            ))}
          </Tabs>
        )}
      </Card>
    </Container>
  );
}

function formatTabName(key) {
  return key.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
}

function isJsonString(str) {
  try {
    JSON.parse(str);
    return true;
  } catch {
    return false;
  }
}

function renderTabContent(key, value) {
  if (value.startsWith("❌")) return <Alert variant="warning">{value}</Alert>;

  if (key === "cover_letter") {
    return <ReactMarkdown>{value}</ReactMarkdown>;
  }

  if (key === "ats" && isJsonString(value)) {
    const parsed = JSON.parse(value);
    return (
      <>
        <h5>ATS Score:{" "}
          <Badge bg={
            parsed["ATS Score"] >= 80 ? "success" :
              parsed["ATS Score"] >= 60 ? "warning" : "danger"
          }>
            {parsed["ATS Score"]}
          </Badge>
        </h5>
        <h6 className="mt-3">✔️ Strengths</h6>
        <ul>{parsed.strengths?.map((s, i) => <li key={i}>{s}</li>)}</ul>
        <h6 className="mt-3">📉 Suggestions</h6>
        <ul>{parsed.suggestions?.map((s, i) => <li key={i}>{s}</li>)}</ul>
      </>
    );
  }

  if ((key === "resume_summary" || key === "interview_questions" || key === "optimized_resume" || key === "jobs") && isJsonString(value)) {
    return (
      <ReactJson
        src={JSON.parse(value)}
        name={false}
        theme="rjv-default"
        collapsed={false}
        enableClipboard={true}
        displayDataTypes={false}
      />
    );
  }

  return <pre className="bg-light p-3 rounded">{value}</pre>;
}
