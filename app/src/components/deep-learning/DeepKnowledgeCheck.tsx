import { deepLearningContent, deepQuizItems } from '@/content/deepLearningContent';

export function DeepKnowledgeCheck() {
  return (
    <section className="deep-panel deep-panel-secondary">
      <div className="deep-section-head">
        <h3 className="deep-section-title">{deepLearningContent.knowledgeCheckTitle}</h3>
        <p className="deep-section-subtitle">Quick recall prompts to reinforce core MLP and CNN intuition.</p>
      </div>
      <div className="deep-quiz-grid">
        {deepQuizItems.map((item) => (
          <article key={item.id} className="deep-quiz-card">
            <p className="deep-quiz-question">{item.question}</p>
            <p className="deep-quiz-answer">{item.answer}</p>
            <p className="deep-quiz-expl">{item.explanation}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
