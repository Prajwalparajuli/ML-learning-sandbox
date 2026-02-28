export interface DeepQuizItem {
  id: string;
  question: string;
  answer: string;
  explanation: string;
}

export const deepLearningContent = {
  hero: {
    eyebrow: 'Deep Learning Studio',
    title: 'Visual-first MLP and CNN learning',
    description:
      'Run local browser inference on bundled MNIST models and inspect each forward-pass step with concise visual explanations.',
  },
  mlp: {
    title: 'MLP Explorer',
    description:
      'Understand how neurons transform features through hidden layers and activation choices before a final decision.',
    notes: [
      'Each input feature becomes an input node.',
      'Hidden nodes combine weighted signals.',
      'Activation controls how strongly a node responds.',
    ],
  },
  cnn: {
    title: 'CNN Explorer',
    description:
      'See how convolutional filters scan local digit strokes, then summarize feature maps into final class probabilities.',
    notes: [
      'Filters scan local image patches, not the whole image at once.',
      'Feature maps emphasize where patterns appear.',
      'Pooling keeps strong signals while shrinking spatial size.',
    ],
  },
  knowledgeCheckTitle: 'Knowledge Check',
  moduleTabs: {
    mlp: 'MLP: Signal mixing and activation logic',
    cnn: 'CNN: Spatial pattern extraction pipeline',
  },
  checkpointTitle: 'Guided Checkpoints',
};

export const deepQuizItems: DeepQuizItem[] = [
  {
    id: 'hidden-layer-purpose',
    question: 'What is the main role of a hidden layer in an MLP?',
    answer: 'To combine inputs into more abstract patterns.',
    explanation: 'Hidden layers transform simple signals into richer representations used by the output layer.',
  },
  {
    id: 'activation-function',
    question: 'Why do we apply activation functions like ReLU?',
    answer: 'To introduce non-linear behavior.',
    explanation: 'Without activation, stacked layers collapse to a linear transform and lose expressive power.',
  },
  {
    id: 'conv-filter',
    question: 'What does a CNN filter do?',
    answer: 'It scans patches to detect specific visual patterns.',
    explanation: 'A filter slides over local regions and outputs stronger values where the pattern matches.',
  },
  {
    id: 'max-pooling',
    question: 'What is the effect of max pooling?',
    answer: 'It downsamples while preserving strong activations.',
    explanation: 'Pooling reduces spatial size and keeps dominant responses for efficiency and robustness.',
  },
  {
    id: 'deep-name',
    question: 'Why is it called deep learning?',
    answer: 'Because it stacks multiple hidden layers.',
    explanation: 'Depth refers to the number of intermediate transformation layers between input and output.',
  },
];
