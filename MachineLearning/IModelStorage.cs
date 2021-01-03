using Microsoft.ML;

namespace MachineLearning
{
   /// <summary>
   /// Interfaccia per i gestori di storage dei modelli
   /// </summary>
   public interface IModelStorage
   {
      #region Methods
      /// <summary>
      /// Funzione di caricamento modello
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <param name="inputSchema">Schema di input del modello</param>
      /// <returns>Il modello</returns>
      ITransformer LoadModel(IMachineLearningContextProvider context, out DataViewSchema inputSchema);
      /// <summary>
      /// Funzione di salvataggio modello
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="model">Modello da salvare</param>
      /// <param name="inputSchema">Schema di input del modello</param>
      void SaveModel(IMachineLearningContextProvider context, ITransformer model, DataViewSchema inputSchema);
      #endregion
   }
}