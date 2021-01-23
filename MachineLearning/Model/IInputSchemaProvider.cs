using Microsoft.ML;

namespace MachineLearning.Model
{
   /// <summary>
   /// Interfaccia per i providers di schema di input
   /// </summary>
   public interface IInputSchemaProvider
   {
      #region Properties
      /// <summary>
      /// Schema di input del modello
      /// </summary>
      DataViewSchema InputSchema { get; }
      #endregion
   }
}
