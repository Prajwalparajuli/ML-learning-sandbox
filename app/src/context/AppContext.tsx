import { createContext, useContext, useReducer, type ReactNode } from "react";

type Model = "linear" | "ridge" | "lasso";
type Dataset = "linear" | "noisy";

interface Params {
  slope: number;
  intercept: number;
  lambda?: number;
}

interface Metrics {
  r2: number;
  rmse: number;
  mae: number;
}

interface State {
  model: Model;
  dataset: Dataset;
  params: Params;
  metrics: Metrics;
}

type Action =
  | { type: "SET_MODEL"; payload: Model }
  | { type: "SET_DATASET"; payload: Dataset }
  | { type: "SET_PARAMS"; payload: Partial<Params> }
  | { type: "SET_METRICS"; payload: Metrics };

const initialState: State = {
  model: "linear",
  dataset: "linear",
  params: { slope: 2, intercept: 1 },
  metrics: { r2: 0, rmse: 0, mae: 0 },
};

const AppContext = createContext<{
  state: State;
  dispatch: React.Dispatch<Action>;
}>({ state: initialState, dispatch: () => null });

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "SET_MODEL":
      return { ...state, model: action.payload };
    case "SET_DATASET":
      return { ...state, dataset: action.payload };
    case "SET_PARAMS":
      return { ...state, params: { ...state.params, ...action.payload } };
    case "SET_METRICS":
      return { ...state, metrics: action.payload };
    default:
      return state;
  }
}

export const AppProvider = ({ children }: { children: ReactNode }) => {
  const [state, dispatch] = useReducer(reducer, initialState);
  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
};

export const useApp = () => useContext(AppContext);
