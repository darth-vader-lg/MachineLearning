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
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="inputSchema">Schema di input del modello</param>
      /// <returns>Il modello</returns>
      ITransformer LoadModel(MachineLearningContext ml, out DataViewSchema inputSchema);
      /// <summary>
      /// Funzione di salvataggio modello
      /// </summary>
      /// <param name="ml">Contesto di machine learning</param>
      /// <param name="model">Modello da salvare</param>
      /// <param name="inputSchema">Schema di input del modello</param>
      void SaveModel(MachineLearningContext ml, ITransformer model, DataViewSchema inputSchema);
      #endregion
   }
}