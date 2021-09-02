using Microsoft.ML;

namespace MachineLearning.Data
{
   /// <summary>
   /// Interfaccia per il filtro di righe di una vista di dati
   /// </summary>
   public interface IDataViewRowFilter
   {
      #region Methods
      /// <summary>
      /// Restituisce lo stato di validita' di una riga  di dati
      /// </summary>
      /// <param name="cursor">Cursore di dati</param>
      /// <returns>true se la riga puntata dal cursore e' valida</returns>
      bool IsValidRow(DataViewRowCursor cursor);
      #endregion
   }

   /// <summary>
   /// Filtro di righe di dati
   /// </summary>
   /// <param name="cursor">Cursore</param>
   /// <returns>true se la riga e' valida</returns>
   public delegate bool DataViewRowFilter(DataViewRowCursor cursor);
}
